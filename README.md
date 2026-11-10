# Koharu

Koharu is a headless manga-page translation service. It detects text regions,
reads Japanese text with PaddleOCR-VL 1.6, translates it with local Gemma 4 E4B
or an explicitly selected hosted provider, removes the original text, and
renders the translation as a PNG. It accepts JPEG, PNG, and WebP pages.

## Container

Build the OCI image and start the service with an NVIDIA GPU and a persistent
cache volume:

```sh
docker build -f Containerfile -t koharu-lean .
docker volume create koharu-cache
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v koharu-cache:/var/cache/koharu koharu-lean
```

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. Koharu
downloads its runtime packages and models into the volume when first needed;
none of the model weights are part of the image. Reuse the volume when
restarting or upgrading the container.

Send a page as the raw request body:

```sh
curl --fail --data-binary @page.webp \
  'http://127.0.0.1:8080/translate?target_language=en' \
  --output page.en.png
```

`POST /translate` returns `image/png`. The `target_language` query parameter
defaults to English; the optional `instructions` parameter accepts URL-encoded
translation guidance. Identical requests are served from a disk cache in the
same volume. The service processes requests one at a time.

## Bare-metal build

On Linux, install Rust 1.99 or later, Clang and libclang, CMake, and the
Fontconfig development headers. For Ubuntu 24.04:

```sh
sudo apt-get install clang libclang-dev cmake libfontconfig-dev
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo build --locked -p koharu-server
./target/debug/koharu-server --listen 127.0.0.1:8080
```

On other systems, set `LIBCLANG_PATH` to the directory containing the
installed `libclang` library as needed. The same `/translate` endpoint is
available without a container.

## Configuration and privacy

Run `koharu-server --help` (or `./target/debug/koharu-server --help`) for the
supported flags, defaults and environment-variable fallbacks. You can choose
the translator, model, inpainting model, listener address, cache directory and
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
