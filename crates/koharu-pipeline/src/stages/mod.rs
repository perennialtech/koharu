mod decensoring;
mod detection;
mod inpainting;
mod ocr;
mod translation;

use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use image::DynamicImage;
use koharu_scene::{
    AssetRole, Edit, EntityId, Generation, Patch, ProducerId, RasterLayer, RasterLayerKind,
    Snapshot,
};

pub use detection::KoharuLayoutRFDetrSeg2XLConfig;

use crate::{Bounds, ImageCache, InpaintingMask, PipelineConfig, RequestTranslation, Stage};

#[derive(Clone)]
pub(crate) struct StageInput {
    scene: koharu_scene::Snapshot,
    page: EntityId,
    entities: Option<Arc<BTreeSet<EntityId>>>,
    region: Option<Bounds>,
    images: Arc<ImageCache>,
    inpainting_mask: Option<InpaintingMask>,
    translation: Option<RequestTranslation>,
}

impl StageInput {
    pub(crate) fn new(
        scene: Snapshot,
        page: EntityId,
        entities: Option<Arc<BTreeSet<EntityId>>>,
        region: Option<Bounds>,
        images: Arc<ImageCache>,
        inpainting_mask: Option<InpaintingMask>,
    ) -> Self {
        Self {
            scene,
            page,
            entities,
            region,
            images,
            inpainting_mask,
            translation: None,
        }
    }

    pub(crate) fn with_translation(mut self, options: Option<RequestTranslation>) -> Self {
        self.translation = options;
        self
    }

    pub(crate) fn page(&self) -> EntityId {
        self.page
    }

    /// The page's decensor layer, once decensoring has repainted the page.
    fn decensor_layer(&self) -> Result<Option<EntityId>> {
        let source = AssetRole::new("source")?;
        for entity in self.scene.children(self.page)? {
            if self
                .scene
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Decensor)
                && self.scene.asset(entity, &source)?.is_some()
            {
                return Ok(Some(entity));
            }
        }
        Ok(None)
    }

    /// The page as stages after decensoring read it: the source with the
    /// decensor layer over it, so text under a censor is found, read and
    /// cleaned from the repainted pixels.
    async fn page_image(&self) -> Result<Option<Arc<DynamicImage>>> {
        let page = self.page;
        let Some(source) = self.images.get(&self.scene, page, "source").await? else {
            return Ok(None);
        };
        let Some(layer) = self.decensor_layer()? else {
            return Ok(Some(source));
        };
        let Some(decensor) = self.images.get(&self.scene, layer, "source").await? else {
            return Ok(Some(source));
        };
        if (decensor.width(), decensor.height()) != (source.width(), source.height()) {
            bail!("decensor layer dimensions do not match page {page}");
        }
        let composite = tokio_rayon::spawn(move || {
            let mut composite = source.to_rgba8();
            image::imageops::overlay(&mut composite, &decensor.to_rgba8(), 0, 0);
            if source.color().has_alpha() {
                DynamicImage::ImageRgba8(composite)
            } else {
                DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(composite).into_rgb8())
            }
        })
        .await;
        Ok(Some(Arc::new(composite)))
    }

    async fn require_page_image(&self) -> Result<Arc<DynamicImage>> {
        let page = self.page;
        self.page_image()
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))
    }

    fn contains_entity(&self, entity: EntityId) -> Result<bool> {
        crate::scope::contains_entity(
            &self.scene,
            self.page,
            self.entities.as_deref(),
            self.region,
            entity,
        )
    }
}

#[async_trait]
trait StageProcessor: Send + Sync {
    fn model(&self) -> &'static str;
    fn skip(&self, _input: &StageInput) -> Result<bool> {
        Ok(false)
    }
    fn unload(&self) -> bool;
    async fn load(&self) -> Result<()>;
    async fn process(&self, input: StageInput) -> Result<Patch>;
}

pub(crate) struct Stages {
    decensoring: decensoring::Processor,
    detection: detection::Processor,
    ocr: ocr::Processor,
    translation: translation::Processor,
    inpainting: inpainting::Processor,
}

impl Stages {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
    ) -> Result<Self> {
        Ok(Self {
            decensoring: decensoring::Processor::new(device.clone()),
            detection: detection::Processor::new(config.detection()?, device.clone()),
            ocr: ocr::Processor::new(config.ocr.clone(), device.clone()),
            translation: translation::Processor::new(config.translation.clone(), translator),
            inpainting: inpainting::Processor::new(config.inpainting()?, device.clone())?,
        })
    }

    fn processor(&self, stage: Stage) -> &dyn StageProcessor {
        match stage {
            Stage::Decensoring => &self.decensoring,
            Stage::Detection => &self.detection,
            Stage::Ocr => &self.ocr,
            Stage::Translation => &self.translation,
            Stage::Inpainting => &self.inpainting,
        }
    }

    pub(crate) fn model(&self, stage: Stage) -> &'static str {
        self.processor(stage).model()
    }

    pub(crate) fn skip(&self, stage: Stage, input: &StageInput) -> Result<bool> {
        self.processor(stage).skip(input)
    }

    pub(crate) async fn load(&self, stage: Stage) -> Result<()> {
        self.processor(stage).load().await
    }

    pub(crate) async fn process(&self, stage: Stage, input: StageInput) -> Result<Patch> {
        self.processor(stage).process(input).await
    }

    pub(crate) fn unload(&self, stage: Stage) -> bool {
        self.processor(stage).unload()
    }
}

fn generation(producer: &str, model: &str) -> Result<Generation> {
    let mut generation = Generation::new(ProducerId::new(producer)?);
    generation.model = Some(model.to_owned());
    Ok(generation)
}

fn finish(edit: Edit) -> Result<Patch> {
    edit.finish().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(bytes: &'static [u8]) -> koharu_scene::AssetInput {
        koharu_scene::AssetInput::new(
            bytes,
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: std::collections::BTreeMap::new(),
            },
        )
    }

    #[tokio::test]
    async fn registry_contains_every_stage() {
        let translator = koharu_translator::Translator::from_config(koharu_config::Config::memory(
            koharu_translator::ProvidersConfig::default(),
        ))
        .unwrap();
        let stages = Stages::new(
            &PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
        )
        .unwrap();

        assert_eq!(
            Stage::ALL.map(|stage| stages.model(stage)),
            [
                "kitsumed-yolo11m-seg+waifu-inpaint-xl",
                "koharu-layout-rfdetr-seg-2xl",
                "paddleocr-vl-1.6",
                "translator",
                "lama",
            ]
        );
    }

    #[tokio::test]
    async fn translation_and_inpainting_compose_without_weakening_text_guards() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("page", 1.0, 1.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let text = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                text,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("before".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        setup
            .set_asset(
                page,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"source"),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let base = session.snapshot();

        let mut text_edit = base.edit();
        text_edit.observe::<koharu_scene::SourceText>(text).unwrap();
        text_edit
            .observe::<koharu_scene::Translation>(text)
            .unwrap();
        text_edit
            .set(
                text,
                &koharu_scene::Translation {
                    text: koharu_scene::Authored::user("after".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let text_patch = text_edit.finish().unwrap();

        let mut image_edit = base.edit();
        image_edit.observe_assets(page).unwrap();
        let cleanup = image_edit
            .add_entity(page, koharu_scene::At::Start)
            .unwrap();
        image_edit
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: koharu_scene::Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        image_edit
            .set_asset(
                cleanup,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"clean"),
            )
            .unwrap();
        let image_patch = image_edit.finish().unwrap();

        let image_first = base.preview([&image_patch]).unwrap();
        assert!(text_patch.rebase_on(&image_first).is_ok());
        let text_first = base.preview([&text_patch]).unwrap();
        assert!(image_patch.rebase_on(&text_first).is_ok());

        let changed_source = base
            .patch(|edit| {
                edit.set(
                    text,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("changed".to_owned()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed_source = base.preview([&changed_source]).unwrap();
        assert!(text_patch.rebase_on(&changed_source).is_err());
    }
}
