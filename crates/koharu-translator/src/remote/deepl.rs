// DeepL API v2. Ported from:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/deepl.rs
// The key now comes from the environment, and keys ending in `:fx` use the
// Developer plan's endpoint.

use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{code, machine_text, send_json};
use crate::{
    Language, Result, TranslationRequest,
    provider::{DEEPL_API_KEY, credential},
};

pub(super) async fn translate(
    client: &Client,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = credential(DEEPL_API_KEY)
        .with_context(|| format!("{DEEPL_API_KEY} is required for deepl"))?;
    let target = code("deepl", request.target_language, target)?;
    let source = request
        .source_language
        .map(|language| code("deepl", language, source))
        .transpose()?;
    let root = if api_key.trim_end().ends_with(":fx") {
        "https://api-free.deepl.com"
    } else {
        "https://api.deepl.com"
    };
    let texts = request
        .segments
        .iter()
        .map(|text| machine_text(text))
        .collect::<Vec<_>>();
    let response: Response = send_json(
        "deepl",
        client
            .post(format!("{root}/v2/translate"))
            .header("Authorization", format!("DeepL-Auth-Key {api_key}"))
            .json(&Body {
                text: &texts,
                target_lang: target,
                source_lang: source,
            }),
    )
    .await?;
    Ok(response
        .translations
        .into_iter()
        .map(|translation| translation.text)
        .collect())
}

#[derive(Serialize)]
struct Body<'a> {
    text: &'a [String],
    target_lang: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_lang: Option<&'a str>,
}

#[derive(Deserialize)]
struct Response {
    translations: Vec<Translation>,
}

#[derive(Deserialize)]
struct Translation {
    text: String,
}

fn target(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified => "ZH-HANS",
        ChineseTraditional => "ZH-HANT",
        English => "EN-US",
        French => "FR",
        Portuguese => "PT-PT",
        BrazilianPortuguese => "PT-BR",
        Spanish => "ES",
        Japanese => "JA",
        Turkish => "TR",
        Russian => "RU",
        Arabic => "AR",
        Korean => "KO",
        Thai => "TH",
        Italian => "IT",
        German => "DE",
        Vietnamese => "VI",
        Malay => "MS",
        Indonesian => "ID",
        Hindi => "HI",
        Polish => "PL",
        Czech => "CS",
        Dutch => "NL",
        Gujarati => "GU",
        Urdu => "UR",
        Telugu => "TE",
        Marathi => "MR",
        Hebrew => "HE",
        Bengali => "BN",
        Bulgarian => "BG",
        Tamil => "TA",
        Ukrainian => "UK",
        Kazakh => "KK",
        Belarusian => "BE",
        Hungarian => "HU",
        Filipino | Khmer | Burmese | Persian | Tibetan | Mongolian | Uyghur | Cantonese => {
            return None;
        }
    })
}

fn source(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified | ChineseTraditional => "ZH",
        English => "EN",
        Portuguese | BrazilianPortuguese => "PT",
        language => target(language)?.split('-').next()?,
    })
}
