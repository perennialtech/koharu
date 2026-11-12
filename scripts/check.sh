#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace --tests
