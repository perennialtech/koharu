mod deepl;
mod deeplx;
mod google;
mod microsoft;
mod openai_compatible;
mod openrouter;

use anyhow::Context;
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::de::DeserializeOwned;

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

pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    if !status.is_success() {
        let lower = text.to_ascii_lowercase();
        if status == StatusCode::TOO_MANY_REQUESTS
            || lower.contains("insufficient_quota")
            || lower.contains("resource_exhausted")
            || lower.contains("rate limit exceeded")
            || lower.contains("credit balance is too low")
            || lower.contains("quota exceeded")
        {
            return Err(Error::QuotaExceeded { provider });
        }
        return Err(Error::Api {
            provider,
            status: status.as_u16(),
            message: text.chars().take(4096).collect(),
        });
    }

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
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
