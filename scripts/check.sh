#!/usr/bin/env bash
# Usage: check.sh [lint|test]. With no argument, runs both; CI runs them as
# separate jobs.
set -euo pipefail

run_lint() {
    cargo fmt --all -- --check
    cargo clippy --workspace -- -D warnings
}

run_tests() {
    cargo test --workspace --tests
}

case "${1:-all}" in
lint) run_lint ;;
test) run_tests ;;
all)
    run_lint
    run_tests
    ;;
*)
    echo "usage: $0 [lint|test]" >&2
    exit 2
    ;;
esac
