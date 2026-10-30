from pathlib import Path

import numpy as np
import pytest
import torch

from drift import sampling as s
from drift.augment import DriftAugment
from drift.bank import FieldBank
from drift.grid import warp
from drift.io import load_nifti
from drift.preprocess import resample, with_spacing
from drift.profile import PartnerConfig, SamplingConfig, SiteProfile
from helpers import FakeRegistration, write_subjects


@pytest.fixture
def bank_2d(tmp_path: Path) -> FieldBank:
    return _make_bank(tmp_path, 2)


def _make_bank(tmp_path: Path, ndim: int) -> FieldBank:
    files = write_subjects(tmp_path, ndim)
    profile = SiteProfile(
        name="test",
        partners=PartnerConfig(k=2, feature_labels=(1, 2)),
        sampling=SamplingConfig(sigma_mm=3.0, t=(0.5, 2.0)),
    )
    bank = FieldBank(tmp_path / "bank", profile=profile)
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    bank.build(registration=FakeRegistration(scale=2.0))
    return bank


def _sample(bank: FieldBank, sid: str = "s1") -> dict[str, object]:
    img, _ = load_nifti(bank.root / bank.subjects[sid].image)
    lab, _ = load_nifti(bank.root / bank.subjects[sid].labels, dtype=torch.int64)  # type: ignore[operator]
    return {"subject": sid, "image": img[0], "label": lab[0]}


@pytest.mark.parametrize("policy", ["global", "regional"])
def test_image_and_labels_move_together(tmp_path: Path, ndim: int, policy: str) -> None:
    bank = _make_bank(tmp_path, ndim)
    pol = (
        s.GlobalPolicy(s.Constant(1.5))
        if policy == "global"
        else s.RegionalPolicy({"halves": s.Region((1, 2), split_axis=0)}, s.Uniform(0.5, 2.0))
    )
    aug = DriftAugment(bank, pol, seed=0, return_field=True)
    sample = _sample(bank)
    out = aug(sample)
    assert out["drift"]["accepted"] and out["drift"]["partner"] in bank.partners("s1")
    assert out["image"].shape == sample["image"].shape  # type: ignore[union-attr]
    assert out["label"].dtype == torch.int64
    assert not torch.equal(out["image"], sample["image"])  # type: ignore[arg-type]
    # the label map warped with the same field as the image
    disp = out["drift_displacement"][None]
    expected = warp(sample["image"][None].double(), disp.double())[0]  # type: ignore[index]
    assert torch.allclose(out["image"].double(), expected, atol=1e-5)
    fg = (out["label"] > 0).float()
    agree = ((out["image"] > 0.55).float() == fg).float().mean()
    assert float(agree) > 0.97


def test_seed_makes_it_deterministic(bank_2d: FieldBank) -> None:
    a = DriftAugment(bank_2d, seed=7)
    b = DriftAugment(bank_2d, seed=7)
    c = DriftAugment(bank_2d, seed=8)
    outs = [aug(_sample(bank_2d)) for aug in (a, b, c)]
    assert torch.equal(outs[0]["image"], outs[1]["image"])
    assert outs[0]["drift"] == outs[1]["drift"]
    assert not torch.equal(outs[0]["image"], outs[2]["image"])
    a.set_seed(7)
    assert torch.equal(a(_sample(bank_2d))["image"], outs[0]["image"])


def test_rejection_falls_back_to_identity(bank_2d: FieldBank) -> None:
    aug = DriftAugment(bank_2d, rules=[s.JacobianBounds(0.999, 1.001)], max_retries=3, seed=0)
    sample = _sample(bank_2d)
    out = aug(sample)
    assert out["drift"]["accepted"] is False and out["drift"]["attempts"] == 3
    assert torch.allclose(out["image"], sample["image"])  # type: ignore[arg-type]
    assert torch.equal(out["label"], sample["label"])  # type: ignore[arg-type]


def test_numpy_batched_inputs_and_valid_mask(bank_2d: FieldBank) -> None:
    sample = _sample(bank_2d)
    np_sample = {
        "subject": "s2",
        "image": sample["image"].numpy()[None],  # type: ignore[union-attr]
        "label": sample["label"].numpy()[None],  # type: ignore[union-attr]
    }
    out = DriftAugment(bank_2d, s.GlobalPolicy(s.Constant(3.0)), valid_mask_key="valid", seed=0)(
        np_sample
    )
    assert isinstance(out["image"], np.ndarray) and out["image"].shape == np_sample["image"].shape
    assert out["label"].dtype == np_sample["label"].dtype
    assert out["valid"].dtype == bool and out["valid"].shape[-2:] == out["image"].shape[-2:]


def test_other_grid_via_space_key(bank_2d: FieldBank) -> None:
    sample = _sample(bank_2d)
    space = bank_2d.subjects["s1"].space
    img, fine = resample(sample["image"][None], space, 0.75)  # type: ignore[index]
    lab, _ = resample(sample["label"][None], space, 0.75, mode="nearest")  # type: ignore[index]
    aug = DriftAugment(bank_2d, s.GlobalPolicy(s.Constant(1.0)), seed=0)
    out = aug({"subject": "s1", "image": img[0], "label": lab[0], "space": fine})
    assert out["image"].shape[1:] == fine.shape and out["drift"]["accepted"]
    with pytest.raises(ValueError, match="does not match"):
        aug({"subject": "s1", "image": img[0], "label": lab[0]})
    assert with_spacing(space, 0.75).shape == fine.shape


def test_skip_and_no_partners(bank_2d: FieldBank) -> None:
    sample = _sample(bank_2d)
    out = DriftAugment(bank_2d, prob=0.0, valid_mask_key="valid", seed=0)(sample)
    assert out["drift"]["skipped"] and out["image"] is sample["image"]
    assert bool(out["valid"].all())
    for p in bank_2d.pairs.values():
        p.passed = False
    out = DriftAugment(bank_2d, seed=0)(sample)
    assert out["drift"]["partner"] is None and torch.equal(out["image"], sample["image"])  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["global", "regional"])
def test_field_spacing_matches_full_resolution(tmp_path: Path, ndim: int, policy: str) -> None:
    # Computing the flow on a coarser grid and upsampling it barely changes a smooth sample.
    bank = _make_bank(tmp_path, ndim)
    pol = (
        s.GlobalPolicy(s.Constant(1.5))
        if policy == "global"
        else s.RegionalPolicy({"halves": s.Region((1, 2), split_axis=0)}, s.Constant(1.5))
    )
    sample = {**_sample(bank), "partner": "s2"}
    fine = DriftAugment(bank, pol, seed=0, return_field=True)(sample)
    coarse = DriftAugment(bank, pol, seed=0, return_field=True, field_spacing=3.0)(sample)
    assert coarse["drift"]["accepted"] and fine["drift"]["accepted"]
    assert coarse["drift_displacement"].shape == fine["drift_displacement"].shape
    diff = (coarse["drift_displacement"] - fine["drift_displacement"]).abs()
    inner = diff[(slice(None),) + (slice(4, -4),) * ndim]
    assert float(inner.mean()) < 0.1  # voxels
    agree = (coarse["label"] == fine["label"]).float().mean()
    assert float(agree) > 0.98
