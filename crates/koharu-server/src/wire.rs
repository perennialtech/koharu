//! Messages between the server and its worker process.
//!
//! The worker's stdin is one end of a Unix socket pair and carries messages
//! both ways, so the worker's stdout and stderr stay free for logs. Each
//! message is a JSON header followed by a binary body, each prefixed with its
//! length as a big-endian `u32`. Only `Translate` (the page) and `Translated`
//! (the PNG) have a non-empty body.

use std::{io, os::fd::AsFd};

use axum::body::Bytes;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::UnixStream,
};

use crate::jobs::{Failure, Input, StageProgress};

/// Server to worker. The server sends the next `Translate` only after the
/// worker has finished the previous one.
#[derive(Serialize, Deserialize)]
pub(crate) enum Request {
    Translate(Input),
    /// Stop the current translation.
    Stop,
}

/// Worker to server. `Ready` is sent once, when the models are loaded. Each
/// translation then gets any number of `Progress` and one of `Translated`,
/// `Stopped` or `Failed`.
#[derive(Serialize, Deserialize)]
pub(crate) enum Reply {
    Ready,
    Progress(StageProgress),
    Translated,
    Stopped,
    Failed(Failure),
}

pub(crate) async fn send(
    stream: &mut (impl AsyncWrite + Unpin),
    message: &impl Serialize,
    body: &[u8],
) -> io::Result<()> {
    let header = serde_json::to_vec(message)?;
    stream.write_u32(length(&header)?).await?;
    stream.write_all(&header).await?;
    stream.write_u32(length(body)?).await?;
    stream.write_all(body).await
}

pub(crate) async fn receive<T: DeserializeOwned>(
    stream: &mut (impl AsyncRead + Unpin),
) -> io::Result<(T, Bytes)> {
    let header = read_frame(stream).await?;
    let body = read_frame(stream).await?;
    Ok((serde_json::from_slice(&header)?, Bytes::from(body)))
}

async fn read_frame(stream: &mut (impl AsyncRead + Unpin)) -> io::Result<Vec<u8>> {
    let mut frame = vec![0; stream.read_u32().await? as usize];
    stream.read_exact(&mut frame).await?;
    Ok(frame)
}

fn length(frame: &[u8]) -> io::Result<u32> {
    u32::try_from(frame.len()).map_err(|_| io::Error::other("message is larger than 4 GiB"))
}

/// The worker's end of the socket pair, which the server passes as stdin.
pub(crate) fn stdin_socket() -> io::Result<UnixStream> {
    let socket = std::os::unix::net::UnixStream::from(io::stdin().as_fd().try_clone_to_owned()?);
    socket.set_nonblocking(true)?;
    UnixStream::from_std(socket)
}
