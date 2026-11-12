// Google Translate through Chrome's page-translation endpoint and through the
// Cloud Translation Basic (v2) API. Both take a list of texts and answer in the
// same order.
//
// The web request shape follows kiss-translator's Google preset:
// https://github.com/fishjar/kiss-translator/blob/ab5a5f4/src/builtin-plugins/google.js
// https://cloud.google.com/translate/docs/reference/rest/v2/translate

use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{code, machine_text, send_json};
use crate::{
    Language, Result, TranslationRequest,
    provider::{GOOGLE_CLOUD_API_KEY, credential},
};

const WEB_URL: &str = "https://translate-pa.googleapis.com/v1/translateHtml";
// The public key Chrome's built-in page translation sends.
const WEB_KEY: &str = "AIzaSyATBXajvzQLTDHEQbcpq0Ihe0vWDHmO520";
const CLOUD_URL: &str = "https://translation.googleapis.com/language/translate/v2";

pub(super) async fn translate_web(
    client: &Client,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let (source, target) = languages("google-web", request)?;
    let texts = request
        .segments
        .iter()
        .map(|text| escape_html(&machine_text(text)))
        .collect::<Vec<_>>();
    let body = json!([[texts, source.unwrap_or("auto"), target], "wt_lib"]);
    let response: Value = send_json(
        "google-web",
        client
            .post(WEB_URL)
            .header("Content-Type", "application/json+protobuf")
            .header("X-Goog-API-Key", WEB_KEY)
            .body(body.to_string()),
    )
    .await?;
    let translations = response
        .get(0)
        .and_then(Value::as_array)
        .context("google-web returned no translations")?;
    Ok(translations
        .iter()
        .map(|text| {
            text.as_str()
                .map(unescape_html)
                .context("google-web returned a non-text translation")
        })
        .collect::<anyhow::Result<_>>()?)
}

pub(super) async fn translate_cloud(
    client: &Client,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = credential(GOOGLE_CLOUD_API_KEY)
        .with_context(|| format!("{GOOGLE_CLOUD_API_KEY} is required for google-cloud"))?;
    let (source, target) = languages("google-cloud", request)?;
    let texts = request
        .segments
        .iter()
        .map(|text| machine_text(text))
        .collect::<Vec<_>>();
    let response: CloudResponse = send_json(
        "google-cloud",
        client
            .post(CLOUD_URL)
            .header("X-Goog-Api-Key", api_key)
            .json(&CloudRequest {
                q: &texts,
                target,
                source,
                format: "text",
            }),
    )
    .await?;
    Ok(response
        .data
        .translations
        .into_iter()
        .map(|translation| translation.translated_text)
        .collect())
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
        ChineseSimplified => "zh-CN",
        ChineseTraditional => "zh-TW",
        English => "en",
        French => "fr",
        Portuguese => "pt-PT",
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
        Filipino => "tl",
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
        Hebrew => "iw",
        Bengali => "bn",
        Bulgarian => "bg",
        Tamil => "ta",
        Ukrainian => "uk",
        Tibetan => "bo",
        Kazakh => "kk",
        Mongolian => "mn",
        Uyghur => "ug",
        Cantonese => "yue",
        Belarusian => "be",
        Hungarian => "hu",
    })
}

fn escape_html(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            character => output.push(character),
        }
    }
    output
}

fn unescape_html(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        output.push_str(&rest[..start]);
        rest = &rest[start..];
        let decoded = rest.find(';').and_then(|end| {
            let entity = &rest[1..end];
            let character = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ => entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                    .map_or_else(
                        || entity.strip_prefix('#').and_then(|n| n.parse().ok()),
                        |hex| u32::from_str_radix(hex, 16).ok(),
                    )
                    .and_then(char::from_u32),
            };
            character.map(|character| (character, end))
        });
        match decoded {
            Some((character, end)) => {
                output.push(character);
                rest = &rest[end + 1..];
            }
            None => {
                output.push('&');
                rest = &rest[1..];
            }
        }
    }
    output.push_str(rest);
    output
}

#[derive(Serialize)]
struct CloudRequest<'a> {
    q: &'a [String],
    target: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'a str>,
    format: &'static str,
}

#[derive(Deserialize)]
struct CloudResponse {
    data: CloudData,
}

#[derive(Deserialize)]
struct CloudData {
    translations: Vec<CloudTranslation>,
}

#[derive(Deserialize)]
struct CloudTranslation {
    #[serde(rename = "translatedText")]
    translated_text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_round_trips_markup_characters_and_entities() {
        let text = "A&B <b> \"x\"";
        assert_eq!(unescape_html(&escape_html(text)), text);
        assert_eq!(
            unescape_html("it&#39;s &#x2026; &quot;ok&quot; &bogus; &"),
            "it's … \"ok\" &bogus; &"
        );
    }
}
