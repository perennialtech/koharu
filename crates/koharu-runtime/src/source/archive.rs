use std::{
    fs::{File, create_dir_all},
    io::copy,
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result};
use fast_glob::glob_match;
use flate2::read::GzDecoder;

use crate::download;

pub(crate) const SHARED_LIBRARIES: [&str; 2] = ["**/*.so", "**/*.so.*"];

/// Downloads and unpacks on the store volume before the directory is published.
pub(crate) async fn install_archive<const N: usize>(
    url: &str,
    stage: Arc<tempfile::TempDir>,
    patterns: [impl AsRef<str> + Send + 'static; N],
) -> Result<()> {
    let (_, name) = url
        .rsplit_once('/')
        .context("archive URL has no filename")?;
    let archive = stage.path().join(name);
    download::fetch(url, &archive).await?;
    tokio::task::spawn_blocking(move || {
        let patterns = patterns.each_ref().map(AsRef::as_ref);
        extract(&archive, stage.path(), &patterns)?;
        std::fs::remove_file(&archive)
            .with_context(|| format!("failed to remove {}", archive.display()))
    })
    .await
    .context("archive extraction task failed")?
}

fn extract(archive: &Path, destination: &Path, patterns: &[&str]) -> Result<()> {
    match archive.extension().and_then(|value| value.to_str()) {
        Some("zip" | "whl" | "nupkg") => unzip(archive, destination, patterns),
        Some("gz") => untar(archive, destination, patterns),
        _ => anyhow::bail!("unsupported archive {}", archive.display()),
    }
}

fn selected(path: &Path, patterns: &[&str]) -> bool {
    let path = path.to_string_lossy();
    patterns
        .iter()
        .any(|pattern| glob_match(pattern, path.as_bytes()))
}

fn unzip(archive: &Path, destination: &Path, patterns: &[&str]) -> Result<()> {
    let mut archive = zip::ZipArchive::new(File::open(archive)?)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let Some(path) = entry.enclosed_name() else {
            continue;
        };
        if entry.is_dir() || !selected(&path, patterns) {
            continue;
        }
        let output = destination.join(path);
        create_dir_all(output.parent().context("archive entry has no parent")?)?;
        copy(&mut entry, &mut File::create(output)?)?;
    }
    Ok(())
}

fn untar(archive: &Path, destination: &Path, patterns: &[&str]) -> Result<()> {
    let mut archive = tar::Archive::new(GzDecoder::new(File::open(archive)?));
    for entry in archive.entries()? {
        let mut entry = entry?;
        if selected(&entry.path()?, patterns) {
            entry.unpack_in(destination)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_are_path_aware() {
        assert!(selected(Path::new("bin/runtime.dll"), &["**/*.dll"]));
        assert!(!selected(Path::new("bin/runtime.so"), &["**/*.dll"]));
    }

    #[tokio::test]
    async fn stages_nuget_on_the_store_volume_and_publishes_only_selected_files() {
        use std::io::{Cursor, Write};
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        const NAME: &str = "runtime.nupkg";
        const LIBRARY: &str = "runtimes/linux-x64/native/runtime.so";
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for filename in [LIBRARY, "runtimes/linux-arm64/native/runtime.so"] {
            zip.start_file(filename, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(b"library").unwrap();
        }
        let bytes = zip.finish().unwrap().into_inner();

        let store = tempfile::tempdir().unwrap();
        let parent = store.path();
        let target = parent.join("onnx");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/{NAME}", listener.local_addr().unwrap());
        let install = crate::Store::directory(
            target.clone(),
            |path| path.join(LIBRARY).is_file() && !path.join(NAME).exists(),
            |stage| async move {
                assert_eq!(stage.path().parent(), Some(parent));
                let archive = stage.path().join(NAME);
                let serve = async {
                    let last = bytes.len() - 1;
                    for (end, body) in [(0, &bytes[..1]), (last, &bytes[..])] {
                        let (mut socket, _) = listener.accept().await?;
                        let mut headers = BufReader::new(&mut socket).lines();
                        while let Some(line) = headers.next_line().await? {
                            if line.is_empty() {
                                break;
                            }
                        }
                        if end == last {
                            assert_eq!(std::fs::metadata(&archive)?.len(), bytes.len() as u64);
                        }
                        let headers = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            bytes.len(),
                            body.len()
                        );
                        socket.write_all(headers.as_bytes()).await?;
                        socket.write_all(body).await?;
                    }
                    Ok::<_, anyhow::Error>(())
                };
                tokio::try_join!(serve, install_archive(&url, stage, [LIBRARY]))?;
                Ok(())
            },
        );
        tokio::time::timeout(std::time::Duration::from_secs(10), install)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(std::fs::read(target.join(LIBRARY)).unwrap(), b"library");
        assert!(!target.join("runtimes/linux-arm64").exists());
        assert!(!target.join(NAME).exists());
    }
}
