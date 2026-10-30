"""The offline nnU-Net example script (examples/nnunet_offline_drift.py) on a tiny fake dataset."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from drift.bank import FieldBank
from drift.profile import PartnerConfig, SiteProfile
from helpers import FakeRegistration, write_subjects

pytest.importorskip("nibabel")

SCRIPT = Path(__file__).parents[1] / "examples" / "nnunet_offline_drift.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("nnunet_offline_drift", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offline_copies_and_leak_free_splits(tmp_path: Path) -> None:
    files = write_subjects(tmp_path, 2, n=5)
    raw = tmp_path / "Dataset001_Test"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for sid, (img, lab) in files.items():
        shutil.copy(img, raw / "imagesTr" / f"{sid}_0000.nii.gz")
        shutil.copy(img, raw / "imagesTr" / f"{sid}_0001.nii.gz")  # second modality
        shutil.copy(lab, raw / "labelsTr" / f"{sid}.nii.gz")
    (raw / "dataset.json").write_text(json.dumps({"numTraining": 5, "file_ending": ".nii.gz"}))
    bank = FieldBank(tmp_path / "bank", profile=SiteProfile(name="t", partners=PartnerConfig(k=2)))
    for sid, (img, lab) in list(files.items())[:4]:  # s4 is not in the bank
        bank.add_subject(sid, img, lab)
    bank.build(registration=FakeRegistration())

    out = tmp_path / "Dataset002_TestDrift"
    stats = _load().augment_dataset(
        raw, tmp_path / "bank", out, copies=2, folds=5, seed=0, device=None
    )
    assert stats == {"cases": 5, "copies": 8, "fallback_identity": 0, "not_in_bank": 1}
    assert json.loads((out / "dataset.json").read_text())["numTraining"] == 13
    assert len(list((out / "imagesTr").glob("*.nii.gz"))) == 2 * 13
    assert (out / "labelsTr" / "s0drift01.nii.gz").exists()

    copies = json.loads((out / "drift_copies.json").read_text())
    splits = json.loads((out / "splits_final.json").read_text())
    original = json.loads((out / "splits_original.json").read_text())
    assert len(splits) == 5
    for s, o in zip(splits, original, strict=True):
        assert s["val"] == o["val"]  # validation: real cases only
        for case, cs in copies.items():
            if case in o["val"]:
                assert not set(cs) & set(s["train"])  # no copies of validation cases in training
            else:
                assert set(cs) <= set(s["train"])
