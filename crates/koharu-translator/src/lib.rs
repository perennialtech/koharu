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

impl Translator {
    pub fn from_config(providers: koharu_config::Config<ProvidersConfig>) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            client: koharu_runtime::http_client()?,
        })
    }

    /// Whether any engine in the list would use the page image.
    #[must_use]
    pub fn uses_image(engines: &[Provider], generation: &GenerationConfig) -> bool {
        generation.vision && engines.iter().any(|engine| engine.is_llm())
    }

    /// Tries each engine in order and returns the first complete translation
    /// with the engine that produced it.
    pub async fn translate(
        &self,
        engines: &[Provider],
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<(Provider, Vec<String>)> {
        let first = engines.first().context("no translation engine selected")?;
        if request.segments.is_empty() {
            return Ok((*first, request.segments));
        }
        if Self::uses_image(engines, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        let providers = self.providers.read()?.clone();
        let mut failures = Vec::new();
        for &engine in engines {
            match self
                .attempt(&providers, engine, &generation, &request)
                .await
            {
                Ok(translated) => return Ok((engine, translated)),
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
        request: &TranslationRequest,
    ) -> Result<Vec<String>> {
        providers.validate(engine)?;
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
        Ok(translated)
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

    #[test]
    fn only_llm_engines_take_the_page_image() {
        let generation = GenerationConfig::default();
        assert!(!Translator::uses_image(&[Provider::GoogleWeb], &generation));
        assert!(Translator::uses_image(
            &[Provider::GoogleWeb, Provider::OpenRouter],
            &generation
        ));
    }
}
