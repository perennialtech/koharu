// DeepL's web translator through a DeepLX server's `POST /translate`, which
// takes a list of texts and answers in the same order. DeepLX keeps up the
// client profile DeepL's anonymous endpoint checks, so Koharu does not.
// https://github.com/OwO-Network/DeepLX/blob/0f2b50d/service/service.go
//
// `/v2/translate` is not used: it reads only the first form text and joins
// JSON lists with newlines, which loses segment boundaries.

use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{code, machine_text, send_json};
use crate::{Language, ProvidersConfig, Result, TranslationRequest};

// DeepL's anonymous endpoint rejects requests whose texts total more than
// 1500 UTF-16 code units, so longer pages are sent in several requests.
const MAX_REQUEST_UNITS: usize = 1500;

pub(super) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let base_url = providers
        .deeplx_base_url
        .as_ref()
        .context("deeplx requires a server URL")?;
    let url = format!("{}/translate", base_url.as_str().trim_end_matches('/'));
    let target = code("deeplx", request.target_language, target)?;
    let source = request
        .source_language
        .map(|language| code("deeplx", language, source))
        .transpose()?;
    let texts = request
        .segments
        .iter()
        .map(|text| machine_text(text))
        .collect::<Vec<_>>();
    let mut translations = Vec::with_capacity(texts.len());
    for batch in batches(&texts) {
        let response: Response = send_json(
            "deeplx",
            client.post(&url).json(&Body {
                text: batch,
                target_lang: target,
                source_lang: source,
            }),
        )
        .await?;
        translations.extend(response.data);
    }
    Ok(translations)
}

fn batches(texts: &[String]) -> Vec<&[String]> {
    let mut batches = Vec::new();
    let (mut start, mut units) = (0, 0);
    for (index, text) in texts.iter().enumerate() {
        let length = text.encode_utf16().count();
        if index > start && units + length > MAX_REQUEST_UNITS {
            batches.push(&texts[start..index]);
            (start, units) = (index, 0);
        }
        units += length;
    }
    if start < texts.len() {
        batches.push(&texts[start..]);
    }
    batches
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
    data: Vec<String>,
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
        Italian => "IT",
        German => "DE",
        Vietnamese => "VI",
        Indonesian => "ID",
        Polish => "PL",
        Czech => "CS",
        Dutch => "NL",
        Hebrew => "HE",
        Bulgarian => "BG",
        Ukrainian => "UK",
        Hungarian => "HU",
        _ => return None,
    })
}

fn source(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified | ChineseTraditional => "ZH",
        English => "EN",
        Portuguese | BrazilianPortuguese => "PT",
        language => target(language)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_stay_under_the_anonymous_limit() {
        let texts = [
            "a".repeat(900),
            "b".repeat(600),
            "c".repeat(1),
            "d".repeat(2000),
        ]
        .map(String::from)
        .to_vec();
        let lengths = batches(&texts)
            .into_iter()
            .map(<[String]>::len)
            .collect::<Vec<_>>();
        assert_eq!(lengths, [2, 1, 1]);
    }
}
