"""End-to-end CLI runs with the torch backend on small 2D images (no NiftyReg needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drift.cli import main
from drift.profile import (
    PartnerConfig,
    QCConfig,
    RegistrationConfig,
    SamplingConfig,
    SiteProfile,
)
from helpers import write_subjects

pytest.importorskip("nibabel")
from drift.io import load_nifti


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path]:
    files = write_subjects(tmp_path, 2, n=3)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for sid, (img, lab) in files.items():
        img.rename(tmp_path / "images" / f"{sid}.nii.gz")
        lab.rename(tmp_path / "labels" / f"{sid}.nii.gz")
    profile = SiteProfile(
        name="cli_test",
        registration=RegistrationConfig(
            backend="torch",
            similarity="ssd",
            cp_spacing=-4.0,
            levels=2,
            max_iter=40,
            device="cpu",
        ),
        partners=PartnerConfig(k=1, method="random"),
        sampling=SamplingConfig(sigma_mm=3.0),
        qc=QCConfig(det_min=0.05, det_max=20.0),
    )
    path = tmp_path / "profile.yaml"
    profile.to_yaml(path)
    return tmp_path, path


def test_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles"]) == 0
    assert "breast_mri" in capsys.readouterr().out


def test_register(setup: tuple[Path, Path]) -> None:
    root, profile = setup
    out = root / "reg"
    rc = main(
        [
            "register",
            str(root / "images/s0.nii.gz"),
            str(root / "images/s1.nii.gz"),
            "--out",
            str(out),
            "--profile",
            str(profile),
        ]
    )
    assert rc == 0
    v, space = load_nifti(out / "velocity.nii.gz")
    assert v.shape == (1, 2, *space.shape)
    qc = json.loads((out / "qc.json").read_text())
    assert qc["backend"] == "torch" and qc["fold_fraction"] == 0.0
    assert (out / "affine.txt").exists()


def test_bank_build_info_sweep_augment(
    setup: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, profile = setup
    bank = str(root / "bank")
    args = [
        "bank",
        "build",
        bank,
        "--images",
        str(root / "images"),
        "--labels",
        str(root / "labels"),
        "--profile",
        str(profile),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "3 pairs planned, 3 to register" in out
    assert main(args) == 0  # resuming does nothing
    assert "0 to register" in capsys.readouterr().out

    assert main(["bank", "info", bank, "--pairs"]) == 0
    info = capsys.readouterr().out
    assert '"pairs": 3' in info and "dice" in info

    from drift.bank import FieldBank

    a, b = next(iter(FieldBank(bank).pairs))
    sweep = root / "sweep"
    assert main(["sweep", bank, a, b, "--t", "0", "1", "--out", str(sweep)]) == 0
    t0, _ = load_nifti(sweep / f"{a}_t+0.00.nii.gz")
    t1, _ = load_nifti(sweep / f"{a}_t+1.00.nii.gz")
    src, _ = load_nifti(root / "images" / f"{a}.nii.gz")
    assert torch.allclose(t0, src, atol=1e-5) and not torch.allclose(t1, src, atol=1e-3)
    assert (sweep / f"{a}_t+1.00_label.nii.gz").exists()

    aug = root / "aug"
    assert main(["augment", bank, a, "--n", "2", "--out", str(aug), "--seed", "1"]) == 0
    meta = json.loads((aug / f"{a}_drift.json").read_text())
    assert len(meta) == 2 and all(m["partner"] == b for m in meta)
    assert (aug / f"{a}_drift001_label.nii.gz").exists()
    assert main(["sweep", bank, a, "nobody", "--out", str(sweep)]) == 1
