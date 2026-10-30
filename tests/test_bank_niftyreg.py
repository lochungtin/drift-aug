"""End-to-end bank + augmentation with real NiftyReg (the bank sign convention)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from drift import sampling as s
from drift.augment import DriftAugment
from drift.bank import FieldBank, dice
from drift.filters import gaussian_smooth
from drift.preprocess import resample_to
from drift.profile import PartnerConfig, RegistrationConfig, SiteProfile
from drift.space import ImageSpace
from helpers import shapes_pair

pytest.importorskip("nibabel")
from drift.io import load_nifti, save_nifti

pytestmark = pytest.mark.niftyreg


@pytest.mark.parametrize("spacing", [None, 3.0])
def test_bank_moves_a_towards_b(tmp_path: Path, spacing: float | None) -> None:
    n = 64
    sq, ci = shapes_pair(2, n, square=13.5, circle=15.5)
    space = ImageSpace.from_spacing((n, n), 1.5)
    for sid, mask in (("square", sq), ("circle", ci)):
        save_nifti(tmp_path / f"{sid}.nii.gz", gaussian_smooth(mask, 1.5).float(), space)
        save_nifti(tmp_path / f"{sid}_lab.nii.gz", mask.to(torch.int16), space)
    profile = SiteProfile(
        name="shapes",
        registration=RegistrationConfig(
            backend="niftyreg",
            similarity="ssd",
            cp_spacing=-4.0 if spacing is None else 6.0,
            bending_energy=0.0005,
            linear_elasticity=0.0,
            spacing=spacing,
        ),
        partners=PartnerConfig(k=1, method="random"),
    )
    bank = FieldBank(tmp_path / "bank", profile=profile)
    for sid in ("square", "circle"):
        bank.add_subject(sid, tmp_path / f"{sid}.nii.gz", tmp_path / f"{sid}_lab.nii.gz")
    pairs = bank.build()
    assert len(pairs) == 2 and all(p.passed for p in pairs)
    qc = bank.pairs[("square", "circle")].qc
    assert qc["dice"] > 0.95 and qc["dice"] > qc["dice_affine"]

    img, _ = load_nifti(tmp_path / "square.nii.gz")
    lab, _ = load_nifti(tmp_path / "square_lab.nii.gz", dtype=torch.int64)
    circle, _ = load_nifti(tmp_path / "circle_lab.nii.gz", dtype=torch.int64)
    # t = 1 reproduces B's shape after affine alignment (the affine is not part of the field)
    affine = np.load(bank.root / bank.pairs[("square", "circle")].field)["registration_affine"]
    target = resample_to(circle, space, space, affine=affine, mode="nearest")
    results = {}
    for t in (0.0, 1.0):
        aug = DriftAugment(bank, s.GlobalPolicy(s.Constant(t)), seed=0)
        out = aug({"subject": "square", "image": img[0], "label": lab[0]})
        assert out["drift"]["partner"] == "circle"
        results[t] = dice(out["label"], target[0])
    assert results[1.0] == pytest.approx(qc["dice"], abs=0.01)
    assert results[0.0] == pytest.approx(qc["dice_affine"], abs=0.01)


def test_niftyreg_with_given_affine_skips_aladin(tmp_path: Path) -> None:
    from drift.registration.niftyreg import NiftyReg, NiftyRegParams

    n = 48
    sq, ci = shapes_pair(2, n, square=10.5, circle=12.0)
    space = ImageSpace.from_spacing((n, n), 1.5)
    save_nifti(tmp_path / "a.nii.gz", gaussian_smooth(sq, 1.5).float(), space)
    save_nifti(tmp_path / "b.nii.gz", gaussian_smooth(ci, 1.5).float(), space)
    given = np.eye(3)
    given[0, 2] = 1.0
    reg = NiftyReg(NiftyRegParams(cp_spacing=-4.0, similarity="ssd"), workdir=tmp_path / "work")
    r = reg.register(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz", affine=given)
    assert np.allclose(r.affine, given)
    work = next((tmp_path / "work").iterdir())
    assert not (work / "aladin_res.nii.gz").exists()
