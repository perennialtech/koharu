FROM rust:1.99-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang libclang-dev libfontconfig-dev cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib \
    CARGO_BUILD_JOBS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CARGO_PROFILE_RELEASE_DEBUG=0 \
    CARGO_PROFILE_RELEASE_STRIP=symbols
RUN cargo build --locked --release -p koharu-server

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libfontconfig1 libvulkan1 libgomp1 \
    libx11-6 libxext6 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /home/koharu --create-home koharu \
    && mkdir -p /var/cache/koharu \
    && chown koharu:koharu /var/cache/koharu

COPY --from=builder /src/target/release/koharu-server /usr/local/bin/koharu-server
ENV XDG_CACHE_HOME=/var/cache \
    HF_XET_CACHE=/var/cache/koharu/xet \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
USER koharu
VOLUME ["/var/cache/koharu"]
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/koharu-server"]
