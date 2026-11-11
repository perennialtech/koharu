# Koharu

Koharu is a headless manga-page translation service. It detects text regions,
reads Japanese text with PaddleOCR-VL 1.6, translates it with local Gemma 4 E4B
or an explicitly selected hosted provider, removes the original text, and
renders the translation as a PNG. It accepts JPEG, PNG, and WebP pages.

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

The default translator runs locally. For a hosted translator, set
`KOHARU_PROVIDER`, `KOHARU_MODEL`, and the appropriate API key in your shell;
Compose passes those variables through if set. For `docker run`, add the
corresponding `--env KOHARU_PROVIDER --env KOHARU_MODEL` and
`--env OPENROUTER_API_KEY` (or `--env OPENAI_COMPATIBLE_API_KEY`) flags. The
optional `KOHARU_OPENAI_COMPATIBLE_BASE_URL` and `KOHARU_INPAINTING` settings
are also available. Do not commit credentials or put them in command-line
values. Run `docker run --rm ghcr.io/perennialtech/koharu:latest --help` for
the complete configuration reference.

## HTTP API

Open [Swagger UI](http://127.0.0.1:8080/docs/) to try the API, or use the
[OpenAPI JSON](http://127.0.0.1:8080/openapi.json) to build a client.

Submit a page as multipart form data to `POST /translations`. The response
includes a job ID and a `Location` header. Poll that URL or subscribe to its
`/events` endpoint for live progress. When the job completes, download the PNG
from the `result_url` in the response. Use the job's `/cancel` endpoint to
request cancellation.

Koharu runs one job at a time. Jobs continue after clients disconnect. Jobs
and their results live in memory and expire after an hour in a terminal
state or immediately on server restart. Each submission runs independently.

## Bare-metal build

On Linux, install Rust 1.99 or later, Clang and libclang, CMake, and the
Fontconfig development headers. For Ubuntu 24.04:

```sh
sudo apt-get install clang libclang-dev cmake libfontconfig-dev
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo build --locked -p koharu-server
./target/debug/koharu-server --listen 127.0.0.1:8080
```

On other systems, set `LIBCLANG_PATH` to the directory containing the
installed `libclang` library as needed. The same HTTP API is available
without a container.

## Configuration and privacy

Run `koharu-server --help` (or `./target/debug/koharu-server --help`) for the
supported flags, defaults and environment-variable fallbacks. You can choose
the translator, model, inpainting model, listener address, and
OpenAI-compatible base URL there.

Local Gemma translation is the default and does not send images or text to a
hosted translation provider. Selecting OpenRouter or an OpenAI-compatible
endpoint is an explicit operator decision. Hosted translation sends page
images and recognized text to that service. Set `OPENROUTER_API_KEY` for
OpenRouter; `OPENAI_COMPATIBLE_API_KEY` is optional for OpenAI-compatible
endpoints. Never put credentials in command arguments or the repository.

## Development

See [AGENTS.md](AGENTS.md) for repository rules and the one command used by CI
to check changes. Keep contributions focused and follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Copyright 2025–2026 Mayo Takanashi and Koharu contributors. This fork is
licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your
option.
