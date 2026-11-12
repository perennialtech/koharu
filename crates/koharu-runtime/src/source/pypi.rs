use anyhow::{Context, Result};
use serde::Deserialize;

use crate::network;

#[derive(Deserialize)]
struct Metadata {
    urls: Vec<Distribution>,
}

#[derive(Deserialize)]
struct Distribution {
    filename: String,
    url: String,
}

pub(crate) async fn wheel(project: &str) -> Result<String> {
    let url = format!("https://pypi.org/pypi/{project}/json");
    let client = network::http()?;
    let metadata: Metadata = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to query {project}"))?
        .error_for_status()?
        .json()
        .await
        .with_context(|| format!("invalid PyPI metadata for {project}"))?;
    metadata
        .urls
        .into_iter()
        .find(|file| {
            file.filename.ends_with(".whl")
                && file.filename.contains("manylinux")
                && file.filename.contains("x86_64")
        })
        .map(|file| file.url)
        .with_context(|| format!("{project} has no compatible wheel"))
}
