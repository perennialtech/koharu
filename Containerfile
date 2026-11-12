FROM scratch AS cargo-cache

FROM rust:1.99-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib \
    CARGO_PROFILE_RELEASE_DEBUG=0 \
    CARGO_PROFILE_RELEASE_STRIP=symbols \
    CARGO_HOME=/cache/cargo \
    CARGO_TARGET_DIR=/cache/target
ARG EXPORT_CARGO_CACHE=false
RUN --mount=type=bind,from=cargo-cache,target=/cache,rw \
    cargo build --locked --release -p koharu-server \
    && install -Dm755 /cache/target/release/koharu-server /out/koharu-server \
    && if [ "$EXPORT_CARGO_CACHE" = true ]; then cp -a /cache /build-cache; fi

FROM scratch AS cache-export
COPY --from=builder /build-cache/ /

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libfontconfig1 libvulkan1 libgomp1 \
    libx11-6 libxext6 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /home/koharu --create-home koharu \
    && mkdir -p /var/cache/koharu \
    && chown koharu:koharu /var/cache/koharu

COPY --from=builder /out/koharu-server /usr/local/bin/koharu-server
ENV XDG_CACHE_HOME=/var/cache \
    HF_XET_CACHE=/var/cache/koharu/xet \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
USER koharu
VOLUME ["/var/cache/koharu"]
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/koharu-server"]
