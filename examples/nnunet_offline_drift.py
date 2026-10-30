"""Offline DRIFT augmentation for nnU-Net v2.

nnU-Net crops and resamples cases itself and trains on random patches, so the simplest correct way
to use DRIFT with it is offline: write ``--copies`` augmented versions of every training case into a
new nnU-Net raw dataset, and give nnU-Net cross-validation splits in which each copy stays with its
source case. Copies only ever appear in the *training* part of a fold (never in validation), so the
validation Dice is measured on real cases only.

Requirements: a field bank whose subject ids are the nnU-Net case ids (build it on the same cases,
e.g. ``drift bank build bank/ --images <preprocessed images named <case>.nii.gz> ...``). The raw
images may be on any grid of the same world frame as the bank's copies (the field is resampled
through world space), so the original ``imagesTr`` files can be used directly.

Usage::

    python examples/nnunet_offline_drift.py \\
        --raw $nnUNet_raw/Dataset101_Breast --bank bank/ \\
        --out $nnUNet_raw/Dataset102_BreastDrift --copies 4 --folds 5

    nnUNetv2_plan_and_preprocess -d 102
    cp $nnUNet_raw/Dataset102_BreastDrift/splits_final.json \\
        $nnUNet_preprocessed/Dataset102_BreastDrift/
    nnUNetv2_train 102 3d_fullres 0   # etc.

Compare against the same nnU-Net setup on Dataset101 (no DRIFT) with the same folds; this script
writes the matching ``splits_final.json`` for the original dataset too (``splits_original.json``).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch

from drift.augment import DriftAugment
from drift.bank import FieldBank
from drift.io import load_nifti, save_nifti

SUFFIX = ".nii.gz"


def case_ids(raw: Path) -> list[str]:
    return sorted(p.name[: -len(SUFFIX)] for p in (raw / "labelsTr").glob(f"*{SUFFIX}"))


def channels(raw: Path, case: str) -> list[Path]:
    return sorted((raw / "imagesTr").glob(f"{case}_[0-9][0-9][0-9][0-9]{SUFFIX}"))


def kfold(cases: list[str], folds: int, seed: int) -> list[dict[str, list[str]]]:
    """Random k-fold split of the original cases (nnU-Net's default is also a seeded k-fold)."""
    order = cases[:]
    random.Random(seed).shuffle(order)
    parts = [order[i::folds] for i in range(folds)]
    return [
        {
            "train": sorted(c for j, p in enumerate(parts) if j != i for c in p),
            "val": sorted(parts[i]),
        }
        for i in range(folds)
    ]


def augment_dataset(
    raw: Path, bank_root: Path, out: Path, copies: int, folds: int, seed: int, device: str | None
) -> dict[str, int]:
    bank = FieldBank(bank_root)
    aug = DriftAugment(bank, seed=seed, device=device, image_keys=("image",), label_keys=("label",))
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = case_ids(raw)
    copies_of: dict[str, list[str]] = {c: [] for c in cases}
    stats = {"cases": len(cases), "copies": 0, "fallback_identity": 0, "not_in_bank": 0}
    for case in cases:
        # originals are copied unchanged
        for ch in channels(raw, case):
            shutil.copy(ch, out / "imagesTr" / ch.name)
        shutil.copy(raw / "labelsTr" / f"{case}{SUFFIX}", out / "labelsTr" / f"{case}{SUFFIX}")
        if case not in bank.subjects or not bank.partners(case):
            stats["not_in_bank"] += 1
            continue
        imgs = [load_nifti(ch) for ch in channels(raw, case)]
        image = torch.cat([x for x, _ in imgs], dim=1)  # all modalities warped together
        space = imgs[0][1]
        label, _ = load_nifti(raw / "labelsTr" / f"{case}{SUFFIX}", dtype=torch.int64)
        for k in range(copies):
            res = aug({"subject": case, "image": image[0], "label": label[0], "space": space})
            if not res["drift"]["accepted"]:
                stats["fallback_identity"] += 1
                continue
            new = f"{case}drift{k:02d}"  # nnU-Net case ids may not contain '_' before the channel
            for c in range(image.shape[1]):
                save_nifti(
                    out / "imagesTr" / f"{new}_{c:04d}{SUFFIX}", res["image"][c : c + 1], space
                )
            save_nifti(out / "labelsTr" / f"{new}{SUFFIX}", res["label"].to(torch.int16), space)
            copies_of[case].append(new)
            stats["copies"] += 1

    meta = json.loads((raw / "dataset.json").read_text())
    meta["numTraining"] = stats["cases"] + stats["copies"]
    meta["description"] = (meta.get("description", "") + " + DRIFT offline augmentation").strip()
    (out / "dataset.json").write_text(json.dumps(meta, indent=1))
    splits = kfold(cases, folds, seed)
    (out / "splits_original.json").write_text(json.dumps(splits, indent=1))
    with_copies = [
        {
            "train": sorted(s["train"] + [x for c in s["train"] for x in copies_of[c]]),
            "val": s["val"],
        }
        for s in splits
    ]
    (out / "splits_final.json").write_text(json.dumps(with_copies, indent=1))
    (out / "drift_copies.json").write_text(json.dumps(copies_of, indent=1))
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", type=Path, required=True, help="nnU-Net raw dataset folder")
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new nnU-Net raw dataset folder")
    p.add_argument("--copies", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device")
    a = p.parse_args(argv)
    print(json.dumps(augment_dataset(a.raw, a.bank, a.out, a.copies, a.folds, a.seed, a.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
