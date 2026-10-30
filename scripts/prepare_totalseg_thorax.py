"""Select thorax scans from TotalSegmentator MRI and merge their masks into one label map.

TotalSegmentator MRI (Zenodo record 14710732, v2.0.0, CC BY-NC-SA 2.0) has one folder per scan
with ``mri.nii.gz`` and ``segmentations/<structure>.nii.gz``. A scan is kept if both lungs and the
heart are present and the lungs total at least ``--min-lung-ml``. Labels follow
``profiles/thorax_mri.yaml``.

    uv run python scripts/prepare_totalseg_thorax.py \\
        data/totalseg_mri/extracted data/totalseg_mri/thorax
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

#: Combined label map: label -> TotalSegmentator structures (later entries win on overlap).
LABELS: dict[int, tuple[str, ...]] = {
    1: ("lung_left",),
    2: ("lung_right",),
    3: ("heart",),
    4: ("aorta",),
    5: ("esophagus",),
    6: ("inferior_vena_cava",),
    7: ("liver",),
    8: ("stomach",),
    9: ("spleen",),
    10: ("vertebrae", "intervertebral_discs"),
    11: ("spinal_cord",),
    12: ("autochthon_left", "autochthon_right"),
    13: ("scapula_left", "scapula_right"),
    14: ("clavicula_left", "clavicula_right"),
    15: ("humerus_left", "humerus_right"),
}
NAMES = {
    1: "lung_left", 2: "lung_right", 3: "heart", 4: "aorta", 5: "esophagus",
    6: "inferior_vena_cava", 7: "liver", 8: "stomach", 9: "spleen", 10: "spine",
    11: "spinal_cord", 12: "back_muscles", 13: "scapula", 14: "clavicula", 15: "humerus",
}  # fmt: skip


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(nib.load(str(path)).dataobj) > 0


def process(scan: Path, out: Path, min_lung_ml: float) -> dict[str, object]:
    sid = scan.name
    row: dict[str, object] = {"id": sid}
    try:
        img = nib.load(str(scan / "mri.nii.gz"))
        voxel_ml = float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000.0
        seg = scan / "segmentations"
        lungs = [load_mask(seg / f"{n}.nii.gz") for n in ("lung_left", "lung_right")]
        heart = load_mask(seg / "heart.nii.gz")
        lung_ml = sum(float(m.sum()) for m in lungs if m is not None) * voxel_ml
        heart_ml = float(heart.sum()) * voxel_ml if heart is not None else 0.0
        row.update(lung_ml=round(lung_ml, 1), heart_ml=round(heart_ml, 1), shape=img.shape)
        ok = all(m is not None and m.any() for m in lungs) and heart_ml > 0
        if not ok or lung_ml < min_lung_ml or len(img.shape) != 3:
            row["status"] = "skipped: not a thorax scan"
            return row
        labels = np.zeros(img.shape, dtype=np.uint8)
        counts = {}
        for value, structures in LABELS.items():
            for name in structures:
                m = load_mask(seg / f"{name}.nii.gz")
                if m is not None:
                    labels[m] = value
            counts[NAMES[value]] = int((labels == value).sum())
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        nib.save(img, str(out / "images" / f"{sid}.nii.gz"))
        lab = nib.Nifti1Image(labels, img.affine, img.header)
        lab.set_data_dtype(np.uint8)
        nib.save(lab, str(out / "labels" / f"{sid}.nii.gz"))
        row["status"] = "ok"
        row["voxels"] = json.dumps(counts)
    except Exception as e:
        row["status"] = f"error: {e}"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="extracted TotalSegmentator MRI folder")
    p.add_argument("out", type=Path)
    p.add_argument("--min-lung-ml", type=float, default=1000.0)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    scans = sorted(d for d in a.src.iterdir() if (d / "mri.nii.gz").exists())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda s: process(s, a.out, a.min_lung_ml), scans))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (a.out / "labels.json").write_text(json.dumps(NAMES, indent=1))
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{ok} thorax scans of {len(rows)}; see {a.out / 'selection.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
