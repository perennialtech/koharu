# Koharu

Koharu is a headless manga-page translation service. It detects text regions,
reads Japanese text with PaddleOCR-VL 1.6, translates it through machine
translation services or hosted LLMs, removes the original text, and renders
the translation as a PNG. It accepts JPEG, PNG, and WebP pages.

## Docker

Install Docker and, on the host, a compatible NVIDIA driver and the NVIDIA
Container Toolkit. Configure Docker's NVIDIA GPU support before starting
Koharu. The image targets `linux/amd64` and is published at
`ghcr.io/perennialtech/koharu:latest` from `master`. Version tags (`v*`) and
commit-SHA tags are also published by CI. To build from source instead, run
`docker build -f Containerfile -t koharu .` and use `koharu` in place of the
published image below.

New GHCR packages are private by default. After the first CI publication, make
the `koharu` package public in GitHub's package settings for anonymous pulls,
or authenticate with `docker login ghcr.io` before using the published image.

### Docker run

```sh
docker volume create koharu-cache
docker run --rm --name koharu --gpus all \
  --publish 127.0.0.1:8080:8080 \
  --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev \
  --mount type=volume,source=koharu-cache,target=/var/cache/koharu \
  ghcr.io/perennialtech/koharu:latest
```

### Docker Compose

The included [`compose.yaml`](compose.yaml) runs the same image and mounts the
same model and runtime cache with the same hardening settings:

```sh
docker compose up -d
docker compose logs -f koharu
```

To build the image from the current checkout instead of pulling it, run
`docker compose up --build -d`. Stop with `docker compose down`; the named
cache volume remains intact. Koharu downloads runtime packages and models on
first use, so the volume must remain writable and persistent across upgrades.
No model weights are included in the image.

### Security and configuration

The server has **no authentication or TLS**. Both examples expose port 8080
only on the host loopback address; keep it behind a trusted reverse proxy with
authentication and TLS if remote access is required. The image runs as an
unprivileged user, with a read-only root filesystem, a temporary `/tmp`, and
capabilities dropped. Do not mount the Docker socket or give the container
`--privileged` access.

Set configuration and API keys as environment variables in your shell;
Compose passes the documented ones through if set. For `docker run`, add an
`--env NAME` flag for each. Do not commit credentials or put them in
command-line values. Run `docker run --rm ghcr.io/perennialtech/koharu:latest
--help` for the complete configuration reference.

## HTTP API

Open [Swagger UI](http://127.0.0.1:8080/docs/) to try the API, or use the
[OpenAPI JSON](http://127.0.0.1:8080/openapi.json) to build a client.

`GET /translators` lists the engines this server can call and its default
order. Submit a page as multipart form data to `POST /translations`, with one
`translator` field per engine to try in order; leave them out to use the
default order. The response
includes a job ID and a `Location` header. Poll that URL or subscribe to its
`/events` endpoint for live progress. When the job completes, download the PNG
from the `result_url` in the response. Use the job's `/cancel` endpoint to
request cancellation.

Koharu runs one job at a time. Jobs continue after clients disconnect. Jobs
and their results live in memory and expire after an hour in a terminal
state or immediately on server restart. Each submission runs independently.

## Bare-metal build

On Linux, install Rust 1.99 or later, Clang and libclang. Fontconfig is loaded
at runtime. For Ubuntu 24.04:

```sh
sudo apt-get install clang libclang-dev libfontconfig1
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo build --locked -p koharu-server
./target/debug/koharu-server --listen 127.0.0.1:8080
```

On other systems, set `LIBCLANG_PATH` to the directory containing the
installed `libclang` library as needed. The same HTTP API is available
without a container.

## Configuration

Run `koharu-server --help` (or `./target/debug/koharu-server --help`) for the
supported flags, defaults, environment-variable fallbacks and the API keys
each translation provider reads. You can choose the translation engines and
their failover order, the models LLM engines use, the inpainting model, the
listener address, and the endpoints of self-hosted translators there. API
callers choose engines by name but never their models.

Every translation engine is a remote service, so recognized text goes to the
engines you select; LLM engines also receive the page image. Without any
configuration, Koharu uses Google Translate's web endpoint and falls back to
Microsoft Translator's Edge endpoint. Neither needs an account.

To use DeepL without an API key, uncomment the optional `deeplx` service in
[`compose.yaml`](compose.yaml), set `KOHARU_DEEPLX_BASE_URL=http://deeplx:1188`
in the environment where you run Compose, and start the stack. DeepLX is
accessible to Koharu over the Compose network but has no published host port.
Set `KOHARU_TRANSLATORS=deeplx` to use it as the default translator.

## Development

See [AGENTS.md](AGENTS.md) for repository rules and the one command used by CI
to check changes. Keep contributions focused.

## License

Copyright 2025–2026 Mayo Takanashi and Koharu contributors. This fork is
licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your
option.
