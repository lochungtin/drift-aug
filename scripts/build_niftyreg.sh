#!/usr/bin/env bash
# Build NiftyReg (CPU + OpenMP) into third_party/niftyreg/install (gitignored).
# tests/conftest.py picks up third_party/niftyreg/install/bin automatically when NIFTYREG_BIN is unset.
# Needs git, a C++ compiler and uv (cmake comes from PyPI via uvx).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/third_party/niftyreg"
REF="${1:-v2.1.2}"
# The checkout may live on a filesystem git considers "dubious"; allow it for these commands only.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
mkdir -p "$DIR"
if [ ! -d "$DIR/src/.git" ]; then
  git clone https://github.com/KCL-BMEIS/niftyreg.git "$DIR/src"
fi
# NiftyReg derives its version from `git describe`, so tags are required.
git -C "$DIR/src" fetch --tags --quiet
git -C "$DIR/src" checkout --quiet "$REF"
uvx --from cmake cmake -S "$DIR/src" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DIR/install" -DUSE_CUDA=OFF -DUSE_OPENMP=ON -DBUILD_TESTING=OFF -DCHECK_GPU=OFF
uvx --from cmake cmake --build "$DIR/build" -j "$(nproc)"
uvx --from cmake cmake --install "$DIR/build"
echo "NiftyReg installed to $DIR/install/bin"
