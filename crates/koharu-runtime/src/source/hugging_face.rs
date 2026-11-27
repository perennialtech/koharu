use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use hf_hub::{HFClient, HFError, RepoType, RepoTypeAny, XetOperation, split_id};

use crate::{Store, download, network};

static CLIENT: OnceLock<HFClient> = OnceLock::new();

fn client() -> anyhow::Result<HFClient> {
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let http = network::http()?;
    let client = HFClient::builder()
        .client(http)
        .cache_enabled(false)
        // Reopening the stream retries the entire transfer, including its requests.
        .retry_max_attempts(0)
        .build()?;
    Ok(CLIENT.get_or_init(|| client).clone())
}

/// An immutable file snapshot hosted by Hugging Face.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HuggingFaceFile<'a> {
    kind: RepoTypeAny,
    repository: &'a str,
    revision: &'a str,
    filename: &'a str,
}

impl<'a> HuggingFaceFile<'a> {
    #[must_use]
    pub const fn pinned(repository: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            kind: RepoTypeAny::Model,
            repository,
            revision,
            filename,
        }
    }

    #[must_use]
    pub const fn pinned_dataset(repository: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            kind: RepoTypeAny::Dataset,
            ..Self::pinned(repository, revision, filename)
        }
    }

    #[must_use]
    pub fn exists(self) -> bool {
        self.path().is_file()
    }

    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> anyhow::Result<PathBuf> {
        Store::file(self.path(), move |stage| async move {
            self.download(&client()?, &stage, network::config()?).await
        })
        .await
    }

    async fn download(
        self,
        client: &HFClient,
        stage: &Path,
        config: network::Config,
    ) -> anyhow::Result<()> {
        let (owner, name) = split_id(self.repository);
        let repository = client.repository(self.kind, owner, name);
        download::receive(
            self.filename,
            stage,
            config,
            || async {
                repository
                    .download_file_stream()
                    .filename(self.filename)
                    .revision(self.revision)
                    .send()
                    .await
            },
            retry_after,
        )
        .await
    }

    fn path(self) -> PathBuf {
        Store::root()
            .join("hugging-face")
            .join(self.kind.plural())
            .join(self.repository.replace(['/', '\\'], "--"))
            .join("snapshots")
            .join(self.revision)
            .join(self.filename)
    }
}

fn retry_after(error: &HFError) -> Option<Duration> {
    let transient = match error {
        HFError::Request { source, .. } => {
            source.is_timeout() || source.is_request() || source.is_body() || source.is_decode()
        }
        HFError::Http { context } => {
            matches!(context.status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504)
        }
        HFError::RateLimited { retry_after, .. } => return Some(retry_after.unwrap_or_default()),
        HFError::Xet {
            operation: XetOperation::StreamDownload,
            source,
        } => {
            // hf-hub erases the Xet error type. These are the transport categories
            // exposed by the pinned Xet client, distinct from auth and local I/O.
            let reason = source.to_string();
            reason.starts_with("Timeout: ")
                || reason
                    .strip_prefix("Network error: ")
                    .is_some_and(|detail| {
                        // Xet also wraps HTTP status errors in its network category.
                        !detail.contains("HTTP status client error (")
                            || detail.contains("HTTP status client error (408 ")
                            || detail.contains("HTTP status client error (429 ")
                    })
        }
        _ => false,
    };
    transient.then_some(Duration::ZERO)
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, future::ready, io};

    use futures::{StreamExt, stream};

    use super::*;

    #[tokio::test]
    async fn interrupted_body_short_eof_and_stall_restart_the_same_activity() {
        let stage = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let name = "retry-test.gguf";
        let mut events = download::subscribe();
        let mut config = network::Config::default();
        config.read_timeout = 1;
        config.max_retries = 3;
        let attempts = Cell::new(0);

        download::receive(
            name,
            &stage,
            config,
            || {
                assert_eq!(std::fs::metadata(&stage).unwrap().len(), 0);
                let attempt = attempts.replace(attempts.get() + 1);
                let body = match attempt {
                    0 => stream::iter([
                        Ok(b"part".as_slice()),
                        Err(HFError::Xet {
                            operation: XetOperation::StreamDownload,
                            source: io::Error::other("Network error: connection reset").into(),
                        }),
                    ])
                    .boxed(),
                    1 => stream::iter([Ok(b"short".as_slice())]).boxed(),
                    2 => stream::pending().boxed(),
                    3 => stream::iter([Ok(b"complete".as_slice())]).boxed(),
                    _ => panic!("unexpected stream attempt"),
                };
                ready(Ok((Some(8), body)))
            },
            retry_after,
        )
        .await
        .unwrap();

        assert_eq!(attempts.get(), 4);
        assert_eq!(std::fs::read(&stage).unwrap(), b"complete");
        let mut started = Vec::new();
        let mut finished = Vec::new();
        while let Ok(event) = events.try_recv() {
            match event {
                download::Event::Started { id, name: label } if label == name => started.push(id),
                download::Event::Finished { id } if started.contains(&id) => finished.push(id),
                download::Event::Failed { name: label, .. } if label == name => {
                    panic!("unexpected failure")
                }
                _ => {}
            }
        }
        assert_eq!(started.len(), 1);
        assert_eq!(finished, started);
    }

    #[tokio::test]
    async fn retries_a_broken_http_body_and_stops_on_unauthorized_forbidden_or_missing() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = HFClient::builder()
            .endpoint(format!("http://{}", listener.local_addr().unwrap()))
            .token("test")
            .retry_max_attempts(0)
            .build()
            .unwrap();
        let stage = tempfile::tempdir().unwrap();
        let server = async {
            for (method, file, status, body) in [
                ("HEAD", "weights", 200, ""),
                ("GET", "weights", 200, "bad"),
                ("HEAD", "weights", 200, ""),
                ("GET", "weights", 200, "good"),
                ("HEAD", "private", 401, ""),
                ("HEAD", "forbidden", 403, ""),
                ("HEAD", "absent", 404, ""),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut headers = BufReader::new(&mut socket).lines();
                assert_eq!(
                    headers.next_line().await.unwrap().unwrap(),
                    format!("{method} /datasets/org/data/resolve/abc123/{file} HTTP/1.1")
                );
                while let Some(line) = headers.next_line().await.unwrap() {
                    if line.is_empty() {
                        break;
                    }
                }
                let response = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Length: 4\r\nConnection: close\r\n\r\n{body}"
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        };
        let downloads = async {
            for name in ["weights", "private", "forbidden", "absent"] {
                let path = stage.path().join(name);
                let result = HuggingFaceFile::pinned_dataset("org/data", "abc123", name)
                    .download(&client, &path, network::Config::default())
                    .await;
                if name == "weights" {
                    result.unwrap();
                    assert_eq!(std::fs::read(path).unwrap(), b"good");
                } else {
                    assert!(result.is_err());
                    assert!(!path.exists());
                }
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(server, downloads);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_local_write_failure_does_not_reopen_the_stream() {
        let root = tempfile::tempdir().unwrap();
        let stage = root.path().join("stage");
        std::os::unix::fs::symlink("/dev/full", &stage).unwrap();
        let attempts = Cell::new(0);
        let result = download::receive(
            "full",
            &stage,
            network::Config::default(),
            || {
                attempts.set(attempts.get() + 1);
                ready(Ok((Some(1), stream::iter([Ok::<_, HFError>(b"x")]))))
            },
            retry_after,
        )
        .await;
        assert_eq!(
            result.unwrap_err().downcast::<io::Error>().unwrap().kind(),
            io::ErrorKind::StorageFull
        );
        assert_eq!(attempts.get(), 1);
        assert!(!stage.exists());
    }
}
