# DRIFT

**D**iffeomorphic **R**egional **I**nterpolation and extrapolation of **F**low **T**rajectories.

Data augmentation for segmentation that moves an image partway along an inter-patient deformation,
stored as a stationary velocity field `v`: `φ = exp(w(x)·v)`, with a global or per-region time `t`
that can interpolate (`t ∈ [0,1]`) or extrapolate. Every sample is a diffeomorphism, and image and
labels are warped together, so labels stay exact.

[![Branch CI](https://github.com/lochungtin/drift-aug/actions/workflows/branch-ci.yml/badge.svg)](https://github.com/lochungtin/drift-aug/actions/workflows/branch-ci.yml)

> Status: pre-release.

## Install

```bash
pip install drift-aug            # core
pip install "drift-aug[io]"      # NIfTI and N4 support (needed by the command line)
```

Other extras: `viz`, `monai`, `torchio`.

## How it works

1. **Build a field bank** once: register each training subject A to a few partners B (affine, then a
   B-spline SVF), keep the non-rigid SVFs and their QC.
2. **Augment** during training: pick a partner, draw a time per region (from A's labels), and warp A's
   image and labels with `exp(-w·v_AB)`: at `t = 1` A takes B's shape, other `t` interpolate or extrapolate.

Anatomy-specific choices (preprocessing, registration mask and parameters, regions and their `t` ranges,
QC bounds) live in site profiles: `breast_mri` and `thorax_mri` are bundled (`drift profiles`).

## Command line

```bash
drift bank build bank/ --images imgs/ --labels labels/ --profile breast_mri --backend torch
drift bank info bank/ --pairs
drift sweep bank/ case01 case07 --t -0.5 0 0.5 1 1.5 --out sweep/
drift augment bank/ case01 --n 8 --out aug/
drift register fixed.nii.gz moving.nii.gz --out reg/
```

## Python

```python
from drift.augment import DriftAugment
from drift.bank import FieldBank

aug = DriftAugment(FieldBank("bank/"), seed=0)  # policy, mode and QC from the bank's profile
sample = aug({"subject": "case01", "image": image, "label": label})  # (C, *spatial) tensors/arrays
```

Framework adapters (apply them before random cropping):

- MONAI: `drift.integrations.monai.DriftAugmentd(["image", "label"], bank)`; reads the grid from the
  `MetaTensor` affine.
- TorchIO: `drift.integrations.torchio.RandomDrift(bank)`; warps `ScalarImage`s and `LabelMap`s.
- nnU-Net: `examples/nnunet_offline_drift.py` writes augmented copies into a new nnU-Net dataset with
  leak-free cross-validation splits.

## Registration backends

- `torch` (default): PyTorch reimplementation (NMI/LNCC/SSD, B-spline SVF, GPU). No external
  dependencies. On labelled breast MRI it matched or beat NiftyReg (Dice 0.924 vs 0.905) at ~4.6× the speed.
- `niftyreg`: wraps [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (`reg_aladin`, `reg_f3d -vel`);
  the reference implementation. Build it with `scripts/build_niftyreg.sh`, or put the binaries on `PATH`
  or in `$NIFTYREG_BIN`.

## Development

```bash
uv sync --extra dev --extra io   # add --all-extras for monai/torchio
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests that need NiftyReg, a GPU or real data skip themselves when those are missing. `scripts/build_niftyreg.sh` builds NiftyReg locally, and CI runs the NiftyReg tests against that build on every push and pull request. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org), which set the next version on release.

## Examples

- `examples/01_square_to_circle_2d.ipynb`: the core idea in 2D.
- `examples/02_regional_edits.ipynb`: per-region times, Eulerian vs Lagrangian.
- `examples/03_build_bank_mri.ipynb`: bank building and augmentation on breast MRI (Duke, set `DRIFT_BREAST_DIR`) or phantoms.
- `examples/nnunet_offline_drift.py`: nnU-Net integration.
