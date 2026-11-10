use std::path::PathBuf;

use koharu_runtime::HuggingFaceFile;

use crate::{ModelGeneration, ModelSelection, QuantizationDefinition};

pub(crate) const DEFAULT_MODEL: &str = "gemma4-e4b-it";
pub(crate) const DEFAULT_QUANTIZATION: &str = "Q4_K_XL";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LocalModelDescriptor {
    pub(crate) id: &'static str,
    pub(crate) reasoning: bool,
    pub(crate) quantizations: &'static [QuantizationDefinition],
    pub(crate) generation: ModelGeneration,
    pub(crate) repository: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) projector: Option<&'static str>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct LocalConfig {}

pub(super) static MODELS: &[LocalModelDescriptor] = &[LocalModelDescriptor {
    id: "gemma4-e4b-it",
    reasoning: true,
    quantizations: &[QuantizationDefinition::new(
        "Q4_K_XL",
        "Q4_K XL",
        "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
    )],
    generation: ModelGeneration {
        temperature: Some(0.2),
        top_k: Some(64),
        top_p: Some(0.9),
        min_p: Some(0.05),
        max_tokens: Some(1000),
        repeat_penalty: Some(1.2),
        frequency_penalty: None,
        presence_penalty: None,
    },
    repository: "unsloth/gemma-4-E4B-it-qat-GGUF",
    revision: "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265",
    projector: Some("mmproj-F16.gguf"),
}];

pub(crate) struct ResolvedLocalModel {
    pub(crate) model: PathBuf,
    pub(crate) projector: Option<PathBuf>,
}

impl LocalModelDescriptor {
    pub(crate) fn file(self, filename: &'static str) -> HuggingFaceFile<'static> {
        HuggingFaceFile::pinned(self.repository, self.revision, filename)
    }

    pub(crate) async fn resolve(
        self,
        selection: &ModelSelection,
    ) -> anyhow::Result<ResolvedLocalModel> {
        let filename = self.filename(selection.quantization.as_deref())?;
        let model = self.file(filename).resolve();
        let projector = async {
            match self.projector {
                Some(filename) => Ok(Some(self.file(filename).resolve().await?)),
                None => Ok(None),
            }
        };
        let (model, projector) = tokio::try_join!(model, projector)?;
        Ok(ResolvedLocalModel { model, projector })
    }

    fn filename(self, quantization: Option<&str>) -> anyhow::Result<&'static str> {
        let quantization = quantization.or_else(|| {
            self.quantizations
                .first()
                .map(|quantization| quantization.id)
        });
        let quantization = quantization
            .ok_or_else(|| anyhow::anyhow!("local model '{}' has no quantizations", self.id))?;
        let definition = self
            .quantizations
            .iter()
            .find(|definition| definition.id == quantization)
            .ok_or_else(|| {
                anyhow::anyhow!("unknown quantization '{quantization}' for '{}'", self.id)
            })?;
        Ok(definition.filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_catalog_has_unique_complete_entries() {
        assert_eq!(MODELS.len(), 1);
        for (index, model) in MODELS.iter().enumerate() {
            let id = model.id;
            assert!(!model.repository.is_empty());
            assert_eq!(model.revision.len(), 40);
            assert!(!model.quantizations.is_empty());
            if let Some(projector) = model.projector {
                assert!(!projector.is_empty());
            }
            for (quantization_index, quantization) in model.quantizations.iter().enumerate() {
                assert!(!quantization.filename.is_empty());
                assert!(
                    !model.quantizations[..quantization_index]
                        .iter()
                        .any(|other| other.id == quantization.id)
                );
            }
            assert!(!MODELS[..index].iter().any(|other| other.id == id));
        }
    }

    #[test]
    fn reasoning_matches_supported_chat_templates() {
        for model in MODELS {
            assert!(model.reasoning, "{} should expose reasoning", model.id);
        }
    }

    #[test]
    fn quantization_selects_its_exact_hugging_face_filename() {
        let descriptor = MODELS
            .iter()
            .find(|descriptor| descriptor.id == "gemma4-e4b-it")
            .expect("Gemma model is cataloged");
        assert_eq!(
            descriptor
                .filename(Some("Q4_K_XL"))
                .expect("quantization is valid"),
            "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
        );
    }
}
