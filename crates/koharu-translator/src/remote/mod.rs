mod deepl;
mod deeplx;
mod google;
mod microsoft;
mod openai_compatible;
mod openrouter;

use anyhow::{Context, bail};
use reqwest::{Client, RequestBuilder, StatusCode, header::HeaderMap};
use serde::{Deserialize, de::DeserializeOwned};

use crate::{
    Error, GenerationConfig, Language, Provider, ProvidersConfig, Result, TranslationRequest,
};

pub(crate) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    provider: Provider,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let model = || {
        providers
            .model(provider)
            .with_context(|| format!("{provider} has no configured model"))
    };
    match provider {
        Provider::GoogleWeb => google::translate_web(client, request).await,
        Provider::GoogleCloud => google::translate_cloud(client, request).await,
        Provider::MicrosoftEdge => microsoft::translate_edge(client, request).await,
        Provider::Azure => microsoft::translate_azure(client, providers, request).await,
        Provider::DeepL => deepl::translate(client, request).await,
        Provider::DeepLx => deeplx::translate(client, providers, request).await,
        Provider::OpenAiCompatible => {
            openai_compatible::translate(client, providers, model()?, generation, request).await
        }
        Provider::OpenRouter => openrouter::translate(client, model()?, generation, request).await,
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    refusal: Option<String>,
}

impl ChatResponse {
    fn translations(self, provider: &str, segments: &[String]) -> anyhow::Result<Vec<String>> {
        let choice = self
            .choices
            .into_iter()
            .next()
            .with_context(|| format!("{provider} returned no choices"))?;
        if let Some(reason) = choice.message.refusal.filter(|text| !text.is_empty()) {
            bail!("{provider} refused the translation: {reason}");
        }
        if let Some(reason) = choice.finish_reason {
            match reason.to_ascii_lowercase().as_str() {
                "stop" | "eos_token" | "end_turn" => {}
                "content_filter" | "safety" | "recitation" => {
                    bail!("{provider} filtered the translation ({reason})")
                }
                "length" | "max_tokens" | "model_length" => {
                    bail!("{provider} truncated the translation ({reason})")
                }
                _ => bail!("{provider} did not complete the translation ({reason})"),
            }
        }
        let text = choice
            .message
            .content
            .with_context(|| format!("{provider} returned no message content"))?;
        crate::prompt::translations(provider, &text, segments)
    }
}

pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;
    let status = response.status();
    let error_headers = (!status.is_success()).then(|| response.headers().clone());
    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    if let Some(headers) = error_headers {
        return Err(api_error(provider, status, &headers, &text));
    }

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
}

fn has_zero_quota_limit(value: &serde_json::Value) -> bool {
    use serde_json::Value;

    match value {
        Value::Object(fields) => fields.iter().any(|(key, value)| {
            (matches!(key.as_str(), "quotaValue" | "limit") && (value == 0 || value == "0"))
                || has_zero_quota_limit(value)
        }),
        Value::Array(items) => items.iter().any(has_zero_quota_limit),
        Value::String(message) => message.split("limit:").skip(1).any(|rest| {
            let mut tokens = rest
                .trim_start()
                .split(|c: char| c.is_whitespace() || c == ',');
            matches!(tokens.next(), Some("0" | "0."))
        }),
        _ => false,
    }
}

fn api_error(provider: &'static str, status: StatusCode, headers: &HeaderMap, text: &str) -> Error {
    let lower = text.to_ascii_lowercase();
    let mut message: String = text.chars().take(4096).collect();
    let mut retry_hint = lower.contains("\"retrydelay\"") || lower.contains("retry in ");
    for name in [
        "retry-after",
        "ratelimit-reset",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ] {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            retry_hint |= !value.trim().is_empty();
            message.push_str(&format!(" ({name}: {value})"));
        }
    }
    let azure_free_quota = provider == "azure"
        && status == StatusCode::FORBIDDEN
        && serde_json::from_str::<serde_json::Value>(text)
            .is_ok_and(|body| body["error"]["code"] == 403001 || body["error"]["code"] == "403001");
    if status == StatusCode::PAYMENT_REQUIRED
        || (provider == "deepl" && status.as_u16() == 456)
        || azure_free_quota
        || (status == StatusCode::TOO_MANY_REQUESTS
            && has_zero_quota_limit(&serde_json::from_str(text).unwrap_or_else(|_| text.into())))
        || [
            "insufficient_quota",
            "billing_not_active",
            "billing_hard_limit_reached",
            "insufficient credits",
            "insufficient_credits",
            "credit balance is too low",
            "out of credits",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return Error::QuotaExceeded { provider, message };
    }
    if status == StatusCode::TOO_MANY_REQUESTS && retry_hint {
        return Error::RateLimited { provider, message };
    }
    if lower.contains("exceeded your current quota") {
        return Error::QuotaExceeded { provider, message };
    }
    if status == StatusCode::TOO_MANY_REQUESTS
        || lower.contains("resource_exhausted")
        || lower.contains("rate limit exceeded")
    {
        return Error::RateLimited { provider, message };
    }
    if lower.contains("quota exceeded") {
        return Error::QuotaExceeded { provider, message };
    }
    Error::Api {
        provider,
        status: status.as_u16(),
        message,
    }
}

/// Maps a language to a provider code, rejecting languages the provider lacks.
pub(super) fn code(
    provider: &'static str,
    language: Language,
    map: fn(Language) -> Option<&'static str>,
) -> Result<&'static str> {
    map(language).ok_or(Error::UnsupportedLanguage { provider, language })
}

/// Text for machine-translation services, which treat each line break as a
/// sentence boundary. OCR breaks vertical text into one line per column, so
/// lines between CJK characters join without a space and others with one.
pub(super) fn machine_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for (index, line) in text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        if index > 0 {
            let previous = output.chars().next_back();
            let next = line.chars().next();
            if !(previous.is_some_and(is_cjk) && next.is_some_and(is_cjk)) {
                output.push(' ');
            }
        }
        output.push_str(line);
    }
    output
}

fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3000}'..='\u{30FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{FF00}'..='\u{FFEF}'
            | '\u{2026}'
            | '\u{2025}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_quota_with_retry_delay_is_temporary() {
        let message = "You exceeded your current quota. Quota exceeded for metric: requests per minute, limit: 10";
        let body = serde_json::json!({"error": {
            "message": message, "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "22s"}]
        }});
        for (headers, text) in [
            (HeaderMap::new(), body.to_string()),
            (
                HeaderMap::from_iter([(reqwest::header::RETRY_AFTER, "22".parse().unwrap())]),
                message.to_owned(),
            ),
        ] {
            assert!(matches!(
                api_error(
                    "openai-compatible",
                    StatusCode::TOO_MANY_REQUESTS,
                    &headers,
                    &text
                ),
                Error::RateLimited { .. }
            ));
        }
    }

    #[test]
    fn gemini_zero_quota_takes_precedence_over_retry_delay() {
        use serde_json::json;

        for (message, violation) in [
            (
                "Quota exceeded for metric: requests, limit: 0, model: gemini",
                json!({}),
            ),
            (
                "You exceeded your current quota",
                json!({"quotaValue": "0"}),
            ),
            ("You exceeded your current quota", json!({"limit": 0})),
        ] {
            let body = json!({"error": {
                "message": message, "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [violation]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "22s"}
                ]
            }});
            assert!(matches!(
                api_error(
                    "openai-compatible",
                    StatusCode::TOO_MANY_REQUESTS,
                    &HeaderMap::new(),
                    &body.to_string()
                ),
                Error::QuotaExceeded { .. }
            ));
        }
    }

    #[test]
    fn azure_free_quota_exhaustion_is_distinct_from_other_forbidden_errors() {
        for (code, quota_exceeded) in [(403001, true), (403000, false)] {
            let body = serde_json::json!({"error": {"code": code}});
            let error = api_error(
                "azure",
                StatusCode::FORBIDDEN,
                &HeaderMap::new(),
                &body.to_string(),
            );
            assert_eq!(matches!(error, Error::QuotaExceeded { .. }), quota_exceeded);
        }
    }

    #[test]
    fn machine_text_joins_vertical_columns_without_spaces() {
        assert_eq!(
            machine_text("お前、まだ\nそんなこと\n言ってるのか？"),
            "お前、まだそんなこと言ってるのか？"
        );
        assert_eq!(machine_text("……\n別に。"), "……別に。");
        assert_eq!(machine_text("NO\nWAY"), "NO WAY");
        assert_eq!(machine_text(" 待って \n\nOK"), "待って OK");
    }
}
