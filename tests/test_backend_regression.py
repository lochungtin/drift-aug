"""Regression test: torch backend against NiftyReg (the reference) on fixed Duke breast-MRI pairs.

torch is the default backend and NiftyReg stays the reference, compared on a fixed
small breast subset. Opt-in (``DRIFT_SLOW=1``, several minutes) and skipped without the data
(``data/duke/{images,labels}``, see data/duke/README.md) or NiftyReg.

Pairs: 002 <- 006 exposed a torch affine failure (L-R scale 0.80 instead of ~1.2; Dice 0.72
vs NiftyReg 0.94), fixed by moment-matching initialisation; 018 <- 025 was the validated pair
where torch trailed NiftyReg most (-0.026). Bounds: per pair, torch Dice >= NiftyReg - 0.05
(validation range -0.027, plus run-to-run noise from non-deterministic GPU histogram
accumulation); on average >= NiftyReg - 0.02; no folds, det J min >= 0.25, better than affine-only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from drift.bank import FieldBank
from drift.io import load_nifti, save_nifti
from drift.preprocess import preprocess
from drift.profile import load_profile

DUKE = Path(__file__).parents[1] / "data" / "duke"
PAIRS = [("Breast_MRI_002", "Breast_MRI_006"), ("Breast_MRI_018", "Breast_MRI_025")]

pytestmark = [pytest.mark.slow, pytest.mark.niftyreg]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Path]]:
    needed = {s for p in PAIRS for s in p}
    if not all((DUKE / d / f"{s}.nii.gz").exists() for s in needed for d in ("images", "labels")):
        pytest.skip("Duke images/labels not available")
    out = tmp_path_factory.mktemp("duke")
    profile = load_profile("breast_mri")
    files = {}
    for sid in sorted(needed):
        img, space = load_nifti(DUKE / "images" / f"{sid}.nii.gz")
        lab, _ = load_nifti(DUKE / "labels" / f"{sid}.nii.gz", dtype=torch.int64)
        img, lab, space = preprocess(
            img, space, labels=lab, **dataclasses.asdict(profile.preprocess)
        )
        assert lab is not None
        save_nifti(out / f"{sid}.nii.gz", img, space)
        save_nifti(out / f"{sid}_lab.nii.gz", lab.to(torch.int16), space)
        files[sid] = (out / f"{sid}.nii.gz", out / f"{sid}_lab.nii.gz")
    return files


def test_torch_matches_niftyreg_on_duke(
    prepared: dict[str, tuple[Path, Path]], tmp_path: Path
) -> None:
    base = load_profile("breast_mri")
    qc = {}
    for backend in ("niftyreg", "torch"):
        profile = dataclasses.replace(
            base, registration=dataclasses.replace(base.registration, backend=backend)
        )
        bank = FieldBank(tmp_path / backend, profile=profile)
        for sid, (img, lab) in prepared.items():
            bank.add_subject(sid, img, lab)
        bank.build(PAIRS)
        qc[backend] = {k: p.qc for k, p in bank.pairs.items()}
    diffs = []
    for pair in PAIRS:
        nr, tr = qc["niftyreg"][pair], qc["torch"][pair]
        assert tr["fold_fraction"] == 0.0 and tr["det_min"] >= 0.25
        assert tr["dice"] > tr["dice_affine"]
        assert tr["dice"] >= nr["dice"] - 0.05, (pair, tr["dice"], nr["dice"])
        diffs.append(tr["dice"] - nr["dice"])
    assert sum(diffs) / len(diffs) >= -0.02, diffs
