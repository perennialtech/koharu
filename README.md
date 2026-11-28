# Koharu

Koharu is a headless manga translation server. Send it a JPEG, PNG, or WebP page over HTTP and get back a translated PNG. Layout detection, OCR, cleanup, and typesetting run locally on an NVIDIA GPU; translation goes to whichever machine-translation service or vision LLM you configure.

## How the pipeline works

Most automatic manga translators draw blunt rectangles over speech bubbles, erase half the surrounding art, misread crowded columns, and stamp tiny low-contrast text back on top. Koharu avoids each of those failure modes:

1. **Dual-detector layout and reading order:** Combines high-resolution RF-DETR Seg 2XL (up to 1920 px) for panels, text masks, and bubbles with ogkalu's RT-DETRv2 comic detector to catch speech bubbles that RF-DETR misses. Duplicate nested bubbles are collapsed, dialogue is matched to its owning balloon, and panels and text blocks are ordered by page gutters so translations read in the right sequence.
2. **Tight source-ink masks and typography measurement:** Fits each mask to the actual source ink instead of wiping whole bounding boxes, keeping crossing line art, screentones, and translucent bubbles intact. From those strokes, Koharu measures writing direction, ink color, outline color, font scale, and tilt angle so slanted lettering follows its real angle instead of snapping to diagonal artifacts.
3. **Neighbor-blanked OCR:** Runs [Hayai OCR v2.7 Supernova](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova) (Japanese, Chinese, Korean, and English) on de-tilted crops after masking out neighboring text regions. Adjacent columns and nearby balloons cannot bleed into the current block's transcription.
4. **Ordered translation failover:** Supports key-free machine translation (`google-web`, `microsoft-edge`), self-hosted `deeplx`, API translators (`deepl`, `google-cloud`, `azure`), and vision-capable LLMs (`openrouter`, `openai-compatible` with strict structured outputs). Engines are tried in order and fail over automatically. LLM engines also receive the page image and optional per-request instructions.
5. **Art-preserving cleanup and legible typesetting:** Plain-paper speech bubbles are filled locally without smearing crossing borders, while free text sitting directly over illustration is removed with LaMa inpainting. Translations are typeset with page-coordinated dialogue sizing, speech-bubble contour fitting, collision avoidance, outline strokes, and WCAG-checked contrast so lettering stays readable over both white bubbles and busy color artwork.

## Docker

Requires Docker, an NVIDIA GPU driver, and the NVIDIA Container Toolkit (`linux/amd64`). Published images live at `ghcr.io/perennialtech/koharu:latest` (plus `v*` and commit-SHA tags). Model weights and managed ML runtimes are downloaded on first use into `/var/cache/koharu`, so mount a persistent writable volume there.

### Docker Compose

[`compose.yaml`](compose.yaml) runs the container unprivileged with a read-only root filesystem, dropped capabilities, and port `8080` bound only to loopback:

```sh
docker compose up -d
docker compose logs -f koharu
```

Build from a local checkout with `docker compose up --build -d`, and stop with `docker compose down` (the `koharu-cache` volume is kept).

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

To build locally instead of pulling, run `docker build -f Containerfile -t koharu .` and replace the image name above with `koharu`.

### Security

The server has **no authentication or TLS**. Keep port `8080` on `127.0.0.1` and put it behind a reverse proxy with authentication and TLS if you need remote access. Pass API keys via environment variables (Compose forwards the documented ones automatically; with `docker run`, pass `--env NAME` without putting secrets on the command line).

## HTTP API

Interactive docs are served at [http://127.0.0.1:8080/docs/](http://127.0.0.1:8080/docs/) with the OpenAPI schema at `/openapi.json`.

| Endpoint | Purpose |
| --- | --- |
| `GET /translators` | List available translation engines and the server's default failover order. |
| `POST /translations` | Submit a multipart form with `image` (JPEG, PNG, or WebP), optional `target_language` (default `en`), optional `instructions` (used by LLM engines), and repeatable `translator` fields for failover order. Returns a job ID and `Location` header. |
| `GET /translations/{id}` | Poll job state and get `result_url` once finished. |
| `GET /translations/{id}/events` | Stream live stage progress over SSE. |
| `GET /translations/{id}/result` | Download the translated PNG. |
| `POST /translations/{id}/cancel` | Request cancellation of a queued or running job. |

Koharu processes one job at a time and continues running after a client disconnects. Jobs and results are held in memory and expire one hour after reaching a terminal state (or on server restart).

## Configuration

Run `koharu-server --help` (or `docker run --rm ghcr.io/perennialtech/koharu:latest --help`) for every flag, environment variable, and provider API key.

With no configuration, Koharu defaults to `google-web` falling back to `microsoft-edge`. Neither requires an account. Every translation engine is a remote service: recognized text is sent to the engine you choose, and LLM engines also receive the page image. Callers can pick engines by name per request, while LLM model selection stays fixed to the server configuration.

For key-free DeepL via DeepLX, uncomment the `deeplx` service in [`compose.yaml`](compose.yaml) (internal to the Compose network with no published host port) and set:

```sh
KOHARU_DEEPLX_BASE_URL=http://deeplx:1188
KOHARU_TRANSLATORS=deeplx
```

## Building from source

On Linux, install Rust 1.99+, Clang, `libclang`, and Fontconfig. On Ubuntu 24.04:

```sh
sudo apt-get install clang libclang-dev libfontconfig1
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo build --locked -p koharu-server
./target/debug/koharu-server --listen 127.0.0.1:8080
```

Set `LIBCLANG_PATH` to your distro's `libclang` directory if it differs.

## Development

See [AGENTS.md](AGENTS.md) for repository rules and `./scripts/check.sh` (the command CI runs).

## License

Copyright 2025–2026 Mayo Takanashi and Koharu contributors. Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
