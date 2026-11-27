//! Translation through machine-translation services and hosted LLMs.

mod backend;
mod error;
mod language;
mod model;
mod prompt;
mod provider;
mod remote;

use anyhow::Context as _;

use error::{Error, Result};

pub use backend::{TranslationContext, TranslationRequest};
pub use language::Language;
pub use model::GenerationConfig;
pub use provider::{Provider, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    client: reqwest::Client,
}

#[derive(Debug)]
pub struct TranslationResult {
    pub engine: Provider,
    pub translations: Vec<String>,
    /// The answering engine consulted the page image, including whether it was absent.
    pub used_image: bool,
}

impl Translator {
    pub fn from_config(providers: koharu_config::Config<ProvidersConfig>) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            client: koharu_runtime::http_client()?,
        })
    }

    /// Whether the engine consults the page image for this generation.
    #[must_use]
    pub fn uses_image(engine: Provider, generation: &GenerationConfig) -> bool {
        generation.vision && engine.is_llm()
    }

    /// Tries each engine in order and returns the first complete translation
    /// with the engine that produced it.
    pub async fn translate(
        &self,
        engines: &[Provider],
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<TranslationResult> {
        let first = engines.first().context("no translation engine selected")?;
        if request.segments.is_empty() {
            return Ok(TranslationResult {
                engine: *first,
                translations: request.segments,
                used_image: false,
            });
        }
        if !generation.vision {
            request.remove_image();
        }

        let providers = self.providers.read()?.clone();
        let mut failures = Vec::new();
        for &engine in engines {
            match self
                .attempt(&providers, engine, &generation, &mut request)
                .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let error = anyhow::Error::from(error);
                    tracing::warn!(%engine, error = format!("{error:#}"), "translation engine failed");
                    failures.push(format!("{engine}: {error:#}"));
                }
            }
        }
        anyhow::bail!("every translation engine failed: {}", failures.join("; "))
    }

    async fn attempt(
        &self,
        providers: &ProvidersConfig,
        engine: Provider,
        generation: &GenerationConfig,
        request: &mut TranslationRequest,
    ) -> Result<TranslationResult> {
        providers.validate(engine)?;
        let used_image = Self::uses_image(engine, generation);
        if used_image {
            request.prepare_image()?;
        }
        let provider: &'static str = engine.into();
        let expected = request.segments.len();
        let translated =
            remote::translate(&self.client, providers, engine, generation, request).await?;
        if translated.len() != expected {
            return Err(Error::SegmentCount {
                provider,
                expected,
                actual: translated.len(),
            });
        }
        if let Some(index) = request
            .segments
            .iter()
            .zip(&translated)
            .position(|(source, text)| !source.trim().is_empty() && text.trim().is_empty())
        {
            return Err(Error::BlankTranslation { provider, index });
        }
        Ok(TranslationResult {
            engine,
            translations: translated,
            used_image,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fails_over_through_every_engine_and_reports_each_failure() {
        let translator =
            Translator::from_config(koharu_config::Config::memory(ProvidersConfig::default()))
                .unwrap();
        let engines = [Provider::DeepLx, Provider::OpenAiCompatible];
        let error = translator
            .translate(
                &engines,
                GenerationConfig::default(),
                TranslationRequest::new(["こんにちは"], Language::English),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("deeplx: deeplx is not configured"),
            "{error}"
        );
        assert!(
            error.contains("openai-compatible: openai-compatible has no model configured"),
            "{error}"
        );
    }

    fn http_response(status_and_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status_and_headers}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn chat_response(choice: serde_json::Value) -> String {
        http_response(
            "200 OK",
            &serde_json::json!({"choices": [choice]}).to_string(),
        )
    }

    async fn translate_responses(
        segments: &[&str],
        replies: &[(&str, String)],
    ) -> anyhow::Result<TranslationResult> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let url: url::Url = format!("http://{}", listener.local_addr()?).parse()?;
        let translator = Translator::from_config(koharu_config::Config::memory(ProvidersConfig {
            openai_compatible_base_url: Some(url.clone()),
            openai_compatible_model: Some("model".into()),
            deeplx_base_url: Some(url),
            ..Default::default()
        }))?;
        let engines = [Provider::OpenAiCompatible, Provider::DeepLx];
        let translate = translator.translate(
            &engines,
            GenerationConfig::default(),
            TranslationRequest::new(segments.iter().copied(), Language::English),
        );
        let serve = async {
            for (path, response) in replies {
                let (socket, _) = listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let mut line = String::new();
                socket.read_line(&mut line).await.unwrap();
                assert_eq!(line, format!("POST {path} HTTP/1.1\r\n"));
                socket
                    .get_mut()
                    .write_all(response.as_bytes())
                    .await
                    .unwrap();
                socket.get_mut().shutdown().await.unwrap();
                tokio::io::copy(&mut socket, &mut tokio::io::sink())
                    .await
                    .unwrap();
            }
        };
        let (result, ()) = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            tokio::join!(translate, serve)
        })
        .await?;
        result
    }

    async fn failover_error(segments: &[&str], first_response: String) -> String {
        let error = translate_responses(
            segments,
            &[
                ("/chat/completions", first_response),
                (
                    "/translate",
                    http_response("400 Bad Request", "fallback failed"),
                ),
            ],
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("fallback failed"), "{error}");
        error
    }

    #[tokio::test]
    async fn hosted_choices_check_refusal_and_finish_before_content() {
        use serde_json::json;

        let content = r#"{"translations":["..."]}"#;
        for (choice, expected) in [
            (
                json!({"message":{"content":null, "refusal":"policy refusal"}}),
                "refused the translation: policy refusal",
            ),
            (
                json!({"message":{"content":content, "refusal":"declined"}, "finish_reason":"stop"}),
                "refused the translation: declined",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"content_filter"}),
                "filtered the translation",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"length"}),
                "truncated the translation",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"MAX_TOKENS"}),
                "truncated the translation",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"model_length"}),
                "truncated the translation (model_length)",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"tool_calls"}),
                "did not complete the translation",
            ),
            (
                json!({"message":{"content":content}, "finish_reason":"unknown_reason"}),
                "did not complete the translation (unknown_reason)",
            ),
        ] {
            let error = failover_error(&["one"], chat_response(choice)).await;
            assert!(error.contains(expected), "{error}");
        }
        for choice in [
            json!({"message":{"content":content}}),
            json!({"message":{"content":content, "refusal":""}, "finish_reason":"stop"}),
            json!({"message":{"content":content}, "finish_reason":"EOS_TOKEN"}),
            json!({"message":{"content":content}, "finish_reason":"end_turn"}),
        ] {
            let result =
                translate_responses(&["one"], &[("/chat/completions", chat_response(choice))])
                    .await
                    .unwrap();
            assert_eq!(result.engine, Provider::OpenAiCompatible);
            assert_eq!(result.translations, ["..."]);
        }
    }

    #[tokio::test]
    async fn rejects_blank_translations_only_for_nonblank_source_segments() {
        let choice = serde_json::json!({"message":{"content":r#"{"translations":[""," \t"]}"#}});
        let error = failover_error(&["", "one"], chat_response(choice.clone())).await;
        assert!(
            error.contains("blank translation for source segment at index 1"),
            "{error}"
        );
        let result = translate_responses(
            &["", "one"],
            &[
                ("/chat/completions", chat_response(choice)),
                (
                    "/translate",
                    http_response("200 OK", r#"{"data":[""," ... "]}"#),
                ),
            ],
        )
        .await
        .unwrap();
        assert_eq!(result.engine, Provider::DeepLx);
        assert_eq!(result.translations, ["", " ... "]);
    }

    #[tokio::test]
    async fn reports_temporary_throttling_and_exhausted_credit_and_fails_over_immediately() {
        for (status, body, diagnostics) in [
            (
                "429 Too Many Requests\r\nRetry-After: 120\r\nX-RateLimit-Reset-Tokens: 2m",
                r#"{"error":{"code":"rate_limit_exceeded"}}"#,
                vec![
                    "temporarily rate limited",
                    "retry-after: 120",
                    "x-ratelimit-reset-tokens: 2m",
                ],
            ),
            (
                "429 Too Many Requests\r\nRetry-After: 120",
                r#"{"error":{"code":"insufficient_quota"}}"#,
                vec!["quota exhausted", "insufficient_quota"],
            ),
            (
                "429 Too Many Requests",
                r#"{"error":{"message":"Quota exceeded for metric: requests, limit: 0, model: gemini","details":[{"retryDelay":"22s"}]}}"#,
                vec!["quota exhausted", "limit: 0"],
            ),
            (
                "402 Payment Required",
                "Insufficient credits",
                vec!["quota exhausted", "Insufficient credits"],
            ),
        ] {
            let error = failover_error(&["one"], http_response(status, body)).await;
            for diagnostic in diagnostics {
                assert!(error.contains(diagnostic), "{error}");
            }
        }
    }

    #[test]
    fn only_llm_engines_take_the_page_image() {
        let generation = GenerationConfig::default();
        assert!(!Translator::uses_image(Provider::GoogleWeb, &generation));
        assert!(Translator::uses_image(Provider::OpenRouter, &generation));
    }
}
