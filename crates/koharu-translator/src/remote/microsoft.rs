// Microsoft Translator through Edge's built-in endpoint and through Azure AI
// Translator v3. Edge's endpoint takes a plain array of strings and answers in
// the Translator v3 response shape.
//
// The Edge request shape follows kiss-translator's Microsoft preset:
// https://github.com/fishjar/kiss-translator/blob/ab5a5f4/src/builtin-plugins/microsoft.js
// https://learn.microsoft.com/azure/ai-services/translator/reference/v3-0-translate

use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{code, machine_text, send_json};
use crate::{
    Language, ProvidersConfig, Result, TranslationRequest,
    provider::{AZURE_API_KEY, credential},
};

const EDGE_URL: &str = "https://edge.microsoft.com/translate/translatetext";
const AZURE_URL: &str = "https://api.cognitive.microsofttranslator.com/translate";

pub(super) async fn translate_edge(
    client: &Client,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let (source, target) = languages("microsoft-edge", request)?;
    let texts = request
        .segments
        .iter()
        .map(|text| machine_text(text))
        .collect::<Vec<_>>();
    let response: Vec<Response> = send_json(
        "microsoft-edge",
        client
            .post(EDGE_URL)
            .query(&[
                ("from", source.unwrap_or("")),
                ("to", target),
                ("isEnterpriseClient", "false"),
            ])
            .json(&texts),
    )
    .await?;
    texts_from(response)
}

pub(super) async fn translate_azure(
    client: &Client,
    providers: &ProvidersConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = credential(AZURE_API_KEY)
        .with_context(|| format!("{AZURE_API_KEY} is required for azure"))?;
    let (source, target) = languages("azure", request)?;
    let body = request
        .segments
        .iter()
        .map(|text| AzureText {
            text: machine_text(text),
        })
        .collect::<Vec<_>>();
    let mut query = vec![("api-version", "3.0"), ("to", target)];
    if let Some(source) = source {
        query.push(("from", source));
    }
    let mut http = client
        .post(AZURE_URL)
        .query(&query)
        .header("Ocp-Apim-Subscription-Key", api_key)
        .json(&body);
    if let Some(region) = providers.azure_region.as_deref() {
        http = http.header("Ocp-Apim-Subscription-Region", region);
    }
    texts_from(send_json("azure", http).await?)
}

fn texts_from(response: Vec<Response>) -> Result<Vec<String>> {
    Ok(response
        .into_iter()
        .map(|item| {
            item.translations
                .into_iter()
                .next()
                .map(|translation| translation.text)
                .context("Microsoft Translator returned an item without a translation")
        })
        .collect::<anyhow::Result<_>>()?)
}

fn languages(
    provider: &'static str,
    request: &TranslationRequest,
) -> Result<(Option<&'static str>, &'static str)> {
    let source = request
        .source_language
        .map(|language| code(provider, language, language_code))
        .transpose()?;
    let target = code(provider, request.target_language, language_code)?;
    Ok((source, target))
}

fn language_code(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified => "zh-Hans",
        ChineseTraditional => "zh-Hant",
        English => "en",
        French => "fr",
        Portuguese => "pt-pt",
        BrazilianPortuguese => "pt",
        Spanish => "es",
        Japanese => "ja",
        Turkish => "tr",
        Russian => "ru",
        Arabic => "ar",
        Korean => "ko",
        Thai => "th",
        Italian => "it",
        German => "de",
        Vietnamese => "vi",
        Malay => "ms",
        Indonesian => "id",
        Filipino => "fil",
        Hindi => "hi",
        Polish => "pl",
        Czech => "cs",
        Dutch => "nl",
        Khmer => "km",
        Burmese => "my",
        Persian => "fa",
        Gujarati => "gu",
        Urdu => "ur",
        Telugu => "te",
        Marathi => "mr",
        Hebrew => "he",
        Bengali => "bn",
        Bulgarian => "bg",
        Tamil => "ta",
        Ukrainian => "uk",
        Tibetan => "bo",
        Kazakh => "kk",
        Mongolian => "mn-Cyrl",
        Uyghur => "ug",
        Cantonese => "yue",
        Hungarian => "hu",
        Belarusian => return None,
    })
}

#[derive(Serialize)]
struct AzureText {
    #[serde(rename = "Text")]
    text: String,
}

#[derive(Deserialize)]
struct Response {
    translations: Vec<Translation>,
}

#[derive(Deserialize)]
struct Translation {
    text: String,
}
