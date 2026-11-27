use super::*;

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        engines: vec![koharu_translator::Provider::OpenAiCompatible],
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let patch = session
        .snapshot()
        .patch(|edit| {
            edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::Skipped { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

struct RejectCommitter;

fn pipeline(translation: TranslationConfig) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}

mod translation {
    use std::{io::Cursor, time::Duration};

    use koharu_scene::{
        AssetInput, AssetMetadata, AssetRole, At, Authored, EntityId, Origin, PageDraft,
        RasterLayer, RasterLayerKind, Session, Snapshot, SourceText, TextLayout, TextLayoutKind,
    };
    use koharu_translator::{Provider, ProvidersConfig, Translator};
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::{TcpListener, TcpStream},
        time::timeout,
    };

    use super::*;
    use crate::stages::{StageInput, Stages};

    fn image_asset(value: u8) -> AssetInput {
        let image = image::GrayImage::from_pixel(1, 1, image::Luma([value]));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        AssetInput::new(
            bytes.into_inner(),
            "image/png",
            AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: Default::default(),
            },
        )
    }

    async fn page_with_text(image: Option<AssetInput>) -> anyhow::Result<(Snapshot, EntityId)> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("page", 1.0, 1.0), At::End)?;
        let content = edit.add_text_content(page, At::End)?;
        edit.set(
            content,
            &SourceText {
                text: Authored::user("source".to_owned()),
                language: None,
            },
        )?;
        edit.add_text_layer(
            page,
            At::End,
            content,
            &TextLayout {
                origin: Origin::User,
                kind: TextLayoutKind::Paragraph,
                angle_degrees: None,
            },
        )?;
        if let Some(image) = image {
            edit.set_asset(page, &AssetRole::new("source")?, image)?;
        }
        let base = session.commit(edit.finish()?).await?.snapshot;
        Ok((base, page))
    }

    fn providers(listener: &TcpListener) -> koharu_config::Config<ProvidersConfig> {
        let url = format!("http://{}", listener.local_addr().unwrap());
        koharu_config::Config::memory(ProvidersConfig {
            openai_compatible_base_url: Some(url.parse().unwrap()),
            openai_compatible_model: Some("model".into()),
            deeplx_base_url: Some(url.parse().unwrap()),
            ..Default::default()
        })
    }

    async fn accept_request(
        listener: &TcpListener,
        path: &str,
    ) -> std::io::Result<BufReader<TcpStream>> {
        let (socket, _) = listener.accept().await?;
        let mut socket = BufReader::new(socket);
        let mut line = String::new();
        socket.read_line(&mut line).await?;
        assert_eq!(line, format!("POST {path} HTTP/1.1\r\n"));
        Ok(socket)
    }

    #[tokio::test]
    async fn stop_wakes_existing_waiters_and_is_visible_to_late_subscribers() {
        let stop = StopToken::default();
        let first = stop.cancelled();
        let second = stop.cancelled();
        tokio::pin!(first, second);
        assert!(futures::poll!(&mut first).is_pending());
        assert!(futures::poll!(&mut second).is_pending());
        stop.stop();
        timeout(Duration::from_secs(2), async {
            tokio::join!(first, second, stop.cancelled());
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn cancelling_stalled_translation_releases_the_lane() -> anyhow::Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let config = PipelineConfig {
            translation: TranslationConfig {
                engines: vec![Provider::OpenAiCompatible, Provider::DeepLx],
                ..Default::default()
            },
            ..Default::default()
        };
        let pipeline = Pipeline::from_config(
            koharu_config::Config::memory(config),
            providers(&listener),
            koharu_ml::Device::cuda(0),
        )?;
        let (base, _) = page_with_text(None).await?;
        let stop = StopToken::default();
        let request = Request {
            operation: Operation::Only {
                stage: Stage::Translation,
            },
            stop: stop.clone(),
            ..Default::default()
        };
        let mut committer = RejectCommitter;
        let run = pipeline.execute(base.clone(), request.clone(), &mut committer);
        tokio::pin!(run);
        let mut stalled = tokio::select! {
            result = &mut run => panic!("translation completed before receiving a response: {result:?}"),
            socket = timeout(Duration::from_secs(10), accept_request(&listener, "/chat/completions")) => socket??,
        };
        stalled
            .get_mut()
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 80\r\n\r\n{")
            .await?;
        assert!(timeout(Duration::from_millis(50), &mut run).await.is_err());
        stop.stop();
        let report = timeout(Duration::from_secs(2), run).await??;
        assert_eq!(report.status, RunStatus::Stopped);
        assert_eq!(report.completed, 0);
        assert_eq!(report.final_revision, base.revision());

        // The same runner must admit another page while the first socket is still open.
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        edit.add_page(PageDraft::new("empty", 1.0, 1.0), At::End)?;
        let empty_page = session.commit(edit.finish()?).await?.snapshot;
        let mut committer = RejectCommitter;
        let next = pipeline.execute(
            empty_page,
            Request {
                stop: StopToken::default(),
                ..request
            },
            &mut committer,
        );
        tokio::select! {
            biased;
            _ = listener.accept() => panic!("unexpected fallback request"),
            result = timeout(Duration::from_secs(2), next) => {
                let report = result??;
                assert_eq!(report.status, RunStatus::Completed);
                assert_eq!(report.completed, 1);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn translation_patch_observes_only_consulted_page_assets() -> anyhow::Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let translator = Translator::from_config(providers(&listener))?;
        let stages = Stages::new(
            &PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
        )?;
        let role = AssetRole::new("source")?;
        let mt = Provider::DeepLx;
        let llm = Provider::OpenAiCompatible;
        let mt_ok = ("/translate", "200 OK", r#"{"data":["text"]}"#);
        let mt_error = ("/translate", "503 Service Unavailable", "");
        let llm_ok = (
            "/chat/completions",
            "200 OK",
            r#"{"choices":[{"message":{"content":"{\"translations\":[\"text\"]}"}}]}"#,
        );
        for (engines, source, responses, observes_image) in [
            (vec![llm], Some(image_asset(1)), vec![llm_ok], true),
            (vec![llm], None, vec![llm_ok], true),
            (vec![mt], Some(image_asset(1)), vec![mt_ok], false),
            (vec![mt, llm], Some(image_asset(1)), vec![mt_ok], false),
            (
                vec![mt, llm],
                Some(image_asset(1)),
                vec![mt_error, llm_ok],
                true,
            ),
        ] {
            let (base, page) = page_with_text(source).await?;
            let input = StageInput::new(base.clone(), page, None, None, Default::default(), None)
                .with_translation(Some(RequestTranslation {
                    target_language: koharu_translator::Language::English,
                    engines: Some(engines),
                    instructions: None,
                }));
            let serve = async {
                for (path, status, body) in responses {
                    let mut socket = accept_request(&listener, path).await?;
                    let response = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    socket.get_mut().write_all(response.as_bytes()).await?;
                    socket.get_mut().shutdown().await?;
                    tokio::io::copy(&mut socket, &mut tokio::io::sink()).await?;
                }
                Ok::<_, std::io::Error>(())
            };
            let (patch, served) = timeout(Duration::from_secs(5), async {
                tokio::join!(stages.process(Stage::Translation, input), serve)
            })
            .await?;
            served?;
            let patch = patch?;
            assert!(!patch.is_empty());
            let cleanup = base.patch(|edit| {
                let layer = edit.add_entity(page, At::Start)?;
                edit.set(
                    layer,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".into(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(layer, &role, image_asset(2))
            })?;
            assert!(base.preview([&cleanup, &patch]).is_ok());
            let replace = base.patch(|edit| edit.set_asset(page, &role, image_asset(3)))?;
            let replaced = base.preview([&replace])?;
            assert_eq!(patch.rebase_on(&replaced).is_err(), observes_image);
        }
        Ok(())
    }
}
