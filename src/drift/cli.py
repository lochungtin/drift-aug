"""Command-line interface.

::

    drift profiles                                    list bundled site profiles
    drift register REF FLO --out DIR                  register one pair (affine + SVF)
    drift bank build ROOT --images DIR [--labels DIR] build (or resume) a field bank
    drift bank info ROOT [--pairs]                    bank summary and per-pair QC
    drift sweep ROOT SUBJECT PARTNER --t ...          A moved towards B at several t
    drift augment ROOT SUBJECT --n N --out DIR        random augmentations of one subject

Images are NIfTI. Subject ids are file names without ``.nii.gz``; label maps are matched by name.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from drift import __version__

log = logging.getLogger("drift")


def _stem(p: Path) -> str:
    name = p.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return p.stem


def _profile(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from drift.profile import load_profile

    profile = load_profile(args.profile)
    reg = profile.registration
    if getattr(args, "backend", None):
        reg = dataclasses.replace(reg, backend=args.backend)
    if getattr(args, "device", None):
        reg = dataclasses.replace(reg, device=args.device)
    profile = dataclasses.replace(profile, registration=reg)
    if getattr(args, "k", None):
        profile = dataclasses.replace(
            profile, partners=dataclasses.replace(profile.partners, k=args.k)
        )
    return profile


# --- commands ----------------------------------------------------------------------------------


def cmd_profiles(args: argparse.Namespace) -> int:
    from drift.profile import available_profiles, load_profile

    for name in available_profiles():
        p = load_profile(name)
        print(f"{name}: {p.description.splitlines()[0] if p.description else ''}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    from drift.io import save_nifti

    profile = _profile(args)
    reg = profile.registration.make(args.threads)
    result = reg.register(
        args.reference, args.floating, reference_mask=args.ref_mask, floating_mask=args.flo_mask
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_nifti(out / "velocity.nii.gz", result.velocity[0], result.space)
    np.savetxt(out / "affine.txt", result.affine)
    js = result.jacobian
    qc = {
        "det_min": float(js.min[0]),
        "det_max": float(js.max[0]),
        "fold_fraction": float(js.fold_fraction[0]),
        **result.qc,
        "velocity_units": "reference voxels; channel i = array axis i; exp(v) pulls the "
        "affinely aligned floating image onto the reference",
        "affine_convention": "affine @ reference_world = floating_world (NiftyReg)",
        "backend": profile.registration.backend,
    }
    (out / "qc.json").write_text(json.dumps(qc, indent=1))
    print(json.dumps({k: v for k, v in qc.items() if not isinstance(v, str)}, indent=1))
    return 0


def _find_images(images: Path, labels: Path | None) -> dict[str, tuple[Path, Path | None]]:
    found = {}
    for p in sorted(images.glob("*.nii*")):
        sid = _stem(p)
        lab = None
        if labels is not None:
            matches = [q for q in labels.glob(f"{sid}.nii*")]
            lab = matches[0] if matches else None
        found[sid] = (p, lab)
    return found


def cmd_bank_build(args: argparse.Namespace) -> int:
    from drift.bank import FieldBank

    profile = _profile(args)
    root = Path(args.root)
    bank = (
        FieldBank(root, profile=profile) if not (root / "bank.json").exists() else FieldBank(root)
    )
    if (root / "bank.json").exists() and (args.backend or args.k):
        log.warning("existing bank: --backend/--k ignored, the bank keeps its stored profile")
    subjects = _find_images(Path(args.images), Path(args.labels) if args.labels else None)
    if args.limit:
        subjects = dict(list(subjects.items())[: args.limit])
    for sid, (img, lab) in subjects.items():
        if sid not in bank.subjects:
            bank.add_subject(sid, img, lab)
            print(f"added {sid}" + ("" if lab else " (no labels)"))
    pairs = bank.plan_pairs(seed=args.seed)
    todo = [p for p in pairs if p not in bank.pairs]
    print(f"{len(pairs)} pairs planned, {len(todo)} to register")

    def report(p):  # type: ignore[no-untyped-def]
        status = "passed" if p.passed else "FAILED " + "; ".join(p.reasons)
        print(f"  {p.a} <- {p.b}: {status}", flush=True)

    bank.build(todo, jobs=args.jobs, callback=report)
    print(json.dumps(bank.summary()))
    return 0


def cmd_bank_info(args: argparse.Namespace) -> int:
    from drift.bank import FieldBank

    bank = FieldBank(args.root)
    print(f"profile: {bank.profile.name} (backend {bank.profile.registration.backend})")
    print(json.dumps(bank.summary()))
    if args.pairs:
        cols = ["det_min", "det_max", "fold_fraction", "max_disp_mm", "dice_affine", "dice"]
        print(f"{'pair':<40}" + "".join(f"{c:>14}" for c in cols) + "  passed")
        for p in bank.pairs.values():
            vals = "".join(f"{p.qc.get(c, float('nan')):>14.3f}" for c in cols)
            print(f"{p.a + ' <- ' + p.b:<40}{vals}  {p.passed} {'; '.join(p.reasons)}")
    return 0


def _subject_arrays(bank, sid: str, image: str | None, label: str | None):  # type: ignore[no-untyped-def]
    from drift.io import load_nifti

    s = bank.subjects[sid]
    img, space = load_nifti(image or bank._abs(s.image))
    lab = None
    lab_path = label or (bank._abs(s.labels) if s.labels else None)
    if lab_path is not None:
        lab, _ = load_nifti(lab_path, dtype=torch.int64)
    return img, lab, space


def cmd_sweep(args: argparse.Namespace) -> int:
    from drift import sampling
    from drift.augment import DriftAugment
    from drift.bank import FieldBank
    from drift.io import save_nifti

    bank = FieldBank(args.root)
    if (args.subject, args.partner) not in bank.pairs:
        print(f"no field {args.subject} <- {args.partner} in the bank", file=sys.stderr)
        return 1
    img, lab, space = _subject_arrays(bank, args.subject, args.image, args.label)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in args.t:
        aug = DriftAugment(
            bank, sampling.GlobalPolicy(sampling.Constant(t)), rules=[], device=args.device
        )
        sample = {"subject": args.subject, "partner": args.partner, "image": img[0], "space": space}
        if lab is not None:
            sample["label"] = lab[0]
        res = aug(sample)
        save_nifti(out / f"{args.subject}_t{t:+.2f}.nii.gz", res["image"], space)
        if lab is not None:
            save_nifti(
                out / f"{args.subject}_t{t:+.2f}_label.nii.gz", res["label"].to(torch.int16), space
            )
        print(f"t = {t:+.2f} -> {out}")
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    from drift.augment import DriftAugment
    from drift.bank import FieldBank
    from drift.io import save_nifti

    bank = FieldBank(args.root)
    img, lab, space = _subject_arrays(bank, args.subject, args.image, args.label)
    aug = DriftAugment(bank, seed=args.seed, device=args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for i in range(args.n):
        sample = {"subject": args.subject, "image": img[0], "space": space}
        if lab is not None:
            sample["label"] = lab[0]
        res = aug(sample)
        name = f"{args.subject}_drift{i:03d}"
        save_nifti(out / f"{name}.nii.gz", res["image"], space)
        if lab is not None:
            save_nifti(out / f"{name}_label.nii.gz", res["label"].to(torch.int16), space)
        meta.append({"name": name, **res["drift"]})
        print(f"{name}: partner {res['drift'].get('partner')}, accepted {res['drift']['accepted']}")
    (out / f"{args.subject}_drift.json").write_text(json.dumps(meta, indent=1))
    return 0


# --- parser ------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("profiles", help="list bundled site profiles")
    sp.set_defaults(func=cmd_profiles)

    def backend_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--profile", default="breast_mri", help="bundled name or YAML path")
        q.add_argument("--backend", choices=["niftyreg", "torch"], help="override the profile")
        q.add_argument("--device", help="torch device for the torch backend")
        q.add_argument("--threads", type=int, help="OpenMP threads for NiftyReg")

    sp = sub.add_parser("register", help="register FLO (moving) to REF (fixed)")
    sp.add_argument("reference")
    sp.add_argument("floating")
    sp.add_argument("--out", required=True)
    sp.add_argument("--ref-mask")
    sp.add_argument("--flo-mask")
    backend_args(sp)
    sp.set_defaults(func=cmd_register)

    bank = sub.add_parser("bank", help="build or inspect a field bank")
    bsub = bank.add_subparsers(dest="bank_command", required=True)
    sp = bsub.add_parser("build", help="add subjects and register partner pairs (resumable)")
    sp.add_argument("root")
    sp.add_argument("--images", required=True, help="directory of (preprocessed) NIfTI images")
    sp.add_argument("--labels", help="directory of label maps with the same file names")
    sp.add_argument("--k", type=int, help="partners per subject (default: profile)")
    sp.add_argument("--jobs", type=int, default=1)
    sp.add_argument("--limit", type=int, help="only the first N subjects")
    sp.add_argument("--seed", type=int, default=0, help="for random partner selection")
    backend_args(sp)
    sp.set_defaults(func=cmd_bank_build)
    sp = bsub.add_parser("info", help="summary and per-pair QC")
    sp.add_argument("root")
    sp.add_argument("--pairs", action="store_true")
    sp.set_defaults(func=cmd_bank_info)

    def sample_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--out", required=True)
        q.add_argument("--image", help="image to warp (default: the bank's copy)")
        q.add_argument("--label", help="label map to warp (default: the bank's copy)")
        q.add_argument("--device")

    sp = sub.add_parser("sweep", help="move SUBJECT towards PARTNER for several t")
    sp.add_argument("root")
    sp.add_argument("subject")
    sp.add_argument("partner")
    sp.add_argument("--t", type=float, nargs="+", default=[-0.5, 0.0, 0.5, 1.0, 1.5])
    sample_args(sp)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("augment", help="random DRIFT augmentations of SUBJECT")
    sp.add_argument("root")
    sp.add_argument("subject")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sample_args(sp)
    sp.set_defaults(func=cmd_augment)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
