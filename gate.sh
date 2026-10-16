#!/usr/bin/env bash
# The build gate. Every check here breaks the build; CI runs exactly this file.
# Usage: ./gate.sh            full gate
#        ./gate.sh --fast     the pre-commit subset (format, lint, types, secrets, tests)
set -euo pipefail
cd "$(dirname "$0")"
fast=0; [[ "${1:-}" == "--fast" ]] && fast=1
step() { printf '\n== %s\n' "$*"; "$@"; }

step uv run ruff format --check .
step uv run ruff check .
step uv run mypy
step uv run basedpyright
step uv run bandit -q -c pyproject.toml -r vesper
mapfile -t tracked < <(git ls-files)
step uv run detect-secrets-hook --baseline .secrets.baseline "${tracked[@]}"
if [[ $fast -eq 0 ]]; then
  # Both extras are audited: the cpu and gpu resolutions pin the same torch release, but the
  # gpu set carries packages the cpu set does not (the CUDA bindings, the NVIDIA libraries,
  # Triton), so an audit of one is not an audit of the other. uv refuses to export two
  # conflicting extras at once, so they are exported one at a time and the union is audited.
  # The wheel index adds a local version label (+cpu, +cu128) that PyPI's advisory database
  # does not know; advisories are filed per upstream version, so the label is dropped. The
  # gate on the GPU box runs its tests against the gpu environment and CI against the cpu one.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  for extra in cpu gpu; do
    uv export --locked --no-emit-project --extra "$extra" --no-hashes -o "$tmp/$extra.txt" >/dev/null
  done
  sed -E 's/\+(cpu|cu[0-9]+)//' "$tmp/cpu.txt" "$tmp/gpu.txt" | grep -v '^#' | sort -u > "$tmp/audit.txt"
  step uv run pip-audit --strict --no-deps --disable-pip -r "$tmp/audit.txt"
fi
step uv run pytest
printf '\nGATE GREEN\n'
