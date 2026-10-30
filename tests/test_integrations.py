"""MONAI and TorchIO adapters (skipped when the framework isn't installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from drift import sampling as s
from drift.bank import FieldBank
from drift.io import load_nifti
from drift.preprocess import resample
from drift.profile import PartnerConfig, SiteProfile
from helpers import FakeRegistration, write_subjects


def _bank(tmp_path: Path) -> FieldBank:
    files = write_subjects(tmp_path, 3)
    bank = FieldBank(tmp_path / "bank", profile=SiteProfile(name="t", partners=PartnerConfig(k=2)))
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    bank.build(registration=FakeRegistration(scale=2.0))
    return bank


def _arrays(bank: FieldBank, sid: str = "s1") -> tuple[torch.Tensor, torch.Tensor]:
    img, _ = load_nifti(bank.root / bank.subjects[sid].image)
    lab, _ = load_nifti(bank.root / str(bank.subjects[sid].labels), dtype=torch.int64)
    return img[0], lab[0]


def test_monai_dictionary_transform(tmp_path: Path) -> None:
    pytest.importorskip("monai")
    from monai.data import MetaTensor
    from monai.transforms import Compose

    from drift.integrations.monai import DriftAugmentd

    bank = _bank(tmp_path)
    img, lab = _arrays(bank)
    affine = torch.tensor(bank.subjects["s1"].space.affine)
    data = {
        "subject": "s1",
        "image": MetaTensor(img, affine=affine),
        "label": MetaTensor(lab, affine=affine),
    }
    t = DriftAugmentd(["image", "label"], bank, s.GlobalPolicy(s.Constant(1.5)))
    out = Compose([t]).set_random_state(seed=3)(data)
    assert isinstance(out["image"], MetaTensor) and isinstance(out["label"], MetaTensor)
    assert torch.equal(out["image"].affine, affine)
    assert out["drift"]["accepted"] and not torch.equal(out["image"].as_tensor(), img)
    assert set(out["label"].unique().tolist()) <= set(lab.unique().tolist())
    again = Compose([t]).set_random_state(seed=3)(data)
    assert torch.equal(again["image"].as_tensor(), out["image"].as_tensor())
    assert again["drift"]["partner"] == out["drift"]["partner"]


def test_monai_reads_the_grid_from_the_affine(tmp_path: Path) -> None:
    pytest.importorskip("monai")
    from monai.data import MetaTensor

    from drift.integrations.monai import DriftAugmentd

    bank = _bank(tmp_path)
    img, _ = _arrays(bank)
    space = bank.subjects["s1"].space
    fine_img, fine = resample(img[None], space, 0.75)
    t = DriftAugmentd(["image"], bank, s.GlobalPolicy(s.Constant(1.0)), seed=0)
    out = t({"subject": "s1", "image": MetaTensor(fine_img[0], affine=torch.tensor(fine.affine))})
    assert out["image"].shape[1:] == fine.shape and out["drift"]["accepted"]


def test_torchio_transform(tmp_path: Path) -> None:
    tio = pytest.importorskip("torchio")
    from drift.integrations.torchio import RandomDrift

    bank = _bank(tmp_path)
    img, lab = _arrays(bank)
    affine = np.asarray(bank.subjects["s1"].space.affine)

    def subject() -> tio.Subject:
        return tio.Subject(
            image=tio.ScalarImage(tensor=img.clone(), affine=affine),
            label=tio.LabelMap(tensor=lab.clone(), affine=affine),
            subject="s1",
        )

    t = RandomDrift(bank, s.GlobalPolicy(s.Constant(1.5)), seed=0)
    out = t(subject())
    assert out["drift"]["accepted"]
    assert not torch.equal(out["image"].data, img)
    assert np.allclose(out["image"].affine, affine)
    assert out["label"].data.dtype == lab.dtype
    # image and label moved together: bright voxels are labelled
    agree = ((out["image"].data > 0.55) == (out["label"].data > 0)).float().mean()
    assert float(agree) > 0.97
    same = RandomDrift(bank, s.GlobalPolicy(s.Constant(1.5)), seed=0)(subject())
    assert torch.equal(same["image"].data, out["image"].data)
    skipped = RandomDrift(bank, p=0.0, seed=0)(subject())
    assert torch.equal(skipped["image"].data, img)
