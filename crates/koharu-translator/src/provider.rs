use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use specta::Type;
use strum::{Display, EnumString, IntoStaticStr, VariantArray};
use url::Url;

/// A translation service Koharu can call.
#[derive(
    Clone,
    Copy,
    Debug,
    Display,
    EnumString,
    Eq,
    Hash,
    IntoStaticStr,
    PartialEq,
    Serialize,
    Deserialize,
    Type,
    VariantArray,
)]
pub enum Provider {
    #[serde(rename = "google-web")]
    #[strum(serialize = "google-web")]
    GoogleWeb,
    #[serde(rename = "microsoft-edge")]
    #[strum(serialize = "microsoft-edge")]
    MicrosoftEdge,
    #[serde(rename = "deeplx")]
    #[strum(serialize = "deeplx")]
    DeepLx,
    #[serde(rename = "deepl")]
    #[strum(serialize = "deepl")]
    DeepL,
    #[serde(rename = "google-cloud")]
    #[strum(serialize = "google-cloud")]
    GoogleCloud,
    #[serde(rename = "azure")]
    #[strum(serialize = "azure")]
    Azure,
    #[serde(rename = "openrouter")]
    #[strum(serialize = "openrouter")]
    OpenRouter,
    #[serde(rename = "openai-compatible")]
    #[strum(serialize = "openai-compatible")]
    OpenAiCompatible,
}

impl Provider {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::GoogleWeb => "Google Translate (web)",
            Self::MicrosoftEdge => "Microsoft Translator (Edge)",
            Self::DeepLx => "DeepL (DeepLX)",
            Self::DeepL => "DeepL API",
            Self::GoogleCloud => "Google Cloud Translation",
            Self::Azure => "Azure AI Translator",
            Self::OpenRouter => "OpenRouter",
            Self::OpenAiCompatible => "OpenAI-compatible",
        }
    }

    /// LLM providers use the server's configured model, follow instructions,
    /// and can read the page image. Machine-translation services do none of
    /// these.
    #[must_use]
    pub const fn is_llm(self) -> bool {
        matches!(self, Self::OpenRouter | Self::OpenAiCompatible)
    }
}

/// Endpoint settings for providers. Credentials come from the environment.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ProvidersConfig {
    /// DeepLX server root, such as `http://deeplx:1188`.
    pub deeplx_base_url: Option<Url>,
    /// OpenAI-compatible endpoint, including its `/v1` path.
    pub openai_compatible_base_url: Option<Url>,
    /// Model the openai-compatible provider uses.
    pub openai_compatible_model: Option<String>,
    /// Model the openrouter provider uses.
    pub openrouter_model: Option<String>,
    /// Azure resource region. Global resources need none.
    pub azure_region: Option<String>,
}

pub(crate) const OPENROUTER_API_KEY: &str = "OPENROUTER_API_KEY";
pub(crate) const OPENAI_COMPATIBLE_API_KEY: &str = "OPENAI_COMPATIBLE_API_KEY";
pub(crate) const DEEPL_API_KEY: &str = "DEEPL_API_KEY";
pub(crate) const GOOGLE_CLOUD_API_KEY: &str = "GOOGLE_CLOUD_TRANSLATION_API_KEY";
pub(crate) const AZURE_API_KEY: &str = "AZURE_TRANSLATOR_KEY";

pub(crate) fn credential(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

impl ProvidersConfig {
    pub fn load() -> Result<koharu_config::Config<Self>> {
        koharu_config::load("providers")
    }

    /// Whether the provider has the endpoint and credentials it needs.
    #[must_use]
    pub fn available(&self, provider: Provider) -> bool {
        match provider {
            Provider::GoogleWeb | Provider::MicrosoftEdge => true,
            Provider::DeepLx => self.deeplx_base_url.is_some(),
            Provider::OpenAiCompatible => {
                self.openai_compatible_base_url.is_some() && self.model(provider).is_some()
            }
            Provider::DeepL => credential(DEEPL_API_KEY).is_some(),
            Provider::GoogleCloud => credential(GOOGLE_CLOUD_API_KEY).is_some(),
            Provider::Azure => credential(AZURE_API_KEY).is_some(),
            Provider::OpenRouter => {
                credential(OPENROUTER_API_KEY).is_some() && self.model(provider).is_some()
            }
        }
    }

    /// Providers that can be used with the current settings.
    #[must_use]
    pub fn available_providers(&self) -> Vec<Provider> {
        Provider::VARIANTS
            .iter()
            .copied()
            .filter(|provider| self.available(*provider))
            .collect()
    }

    /// The configured model of an LLM provider.
    #[must_use]
    pub fn model(&self, provider: Provider) -> Option<&str> {
        match provider {
            Provider::OpenRouter => self.openrouter_model.as_deref(),
            Provider::OpenAiCompatible => self.openai_compatible_model.as_deref(),
            _ => None,
        }
        .filter(|model| !model.is_empty())
    }

    /// Fails when the provider lacks the endpoint, model or credentials it needs.
    pub fn validate(&self, provider: Provider) -> Result<()> {
        ensure!(
            !provider.is_llm() || self.model(provider).is_some(),
            "{provider} has no model configured on this server"
        );
        ensure!(
            self.available(provider),
            "{provider} is not configured on this server"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_providers_need_a_server_model() {
        let mut config = ProvidersConfig {
            openai_compatible_base_url: Some(Url::parse("http://localhost:11434/v1").unwrap()),
            ..ProvidersConfig::default()
        };
        assert!(config.validate(Provider::GoogleWeb).is_ok());
        assert!(config.validate(Provider::OpenAiCompatible).is_err());
        config.openai_compatible_model = Some("model".to_owned());
        assert!(config.validate(Provider::OpenAiCompatible).is_ok());
        assert!(config.validate(Provider::DeepLx).is_err());
        assert!("local".parse::<Provider>().is_err());
    }
}
