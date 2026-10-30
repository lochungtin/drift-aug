"""Field bank: pairwise inter-patient SVFs with per-pair QC, stored on disk.

For each subject A and partner B, B (floating) is registered to A (reference). The non-rigid SVF
``v_AB`` lives on A's grid and ``exp(v_AB)`` pulls the affinely aligned B back onto A. To move A
towards B's shape the augmentation samples ``A ∘ exp(-t·v_AB)``; that sign is applied inside
:meth:`FieldBank.velocity` and never exposed.

Layout::

    root/bank.json              subjects, pairs, QC, the site profile
    root/fields/<a>/<b>.npz     SVF (float16, voxel units of its storage grid) + that grid
    root/masks/<id>.nii.gz      registration masks (when the profile asks for one)

Fields are smooth, so they can be stored on a coarser grid (``profile.store_spacing``); they are
resampled onto whatever grid they are requested on.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from drift.filters import gaussian_smooth
from drift.flow import exp
from drift.grid import warp_labels
from drift.io import PathLike, load_nifti, save_nifti
from drift.jacobian import jacobian_stats
from drift.preprocess import (
    dilate,
    foreground_mask,
    label_mask,
    resample,
    resample_field,
    resample_to,
    with_spacing,
)
from drift.profile import SiteProfile
from drift.registration.base import Registration, RegistrationResult
from drift.space import ImageSpace

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class Subject:
    id: str
    image: str
    labels: str | None
    space: ImageSpace
    feature: float | None = None


@dataclass
class Pair:
    a: str
    b: str
    kind: Literal["inter", "intra"]
    field: str
    qc: dict[str, float] = dataclasses.field(default_factory=dict)
    passed: bool = True
    reasons: list[str] = dataclasses.field(default_factory=list)


def _space_to_json(space: ImageSpace) -> dict[str, Any]:
    return {"shape": list(space.shape), "affine": np.asarray(space.affine).tolist()}


def _space_from_json(d: Mapping[str, Any]) -> ImageSpace:
    return ImageSpace(tuple(d["shape"]), np.asarray(d["affine"], dtype=np.float64))


def dice(a: Tensor, b: Tensor) -> float:
    """Dice of two binary masks."""
    a, b = a != 0, b != 0
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * int((a & b).sum()) / denom


def select_partners(
    features: Mapping[str, float | None],
    k: int,
    method: Literal["random", "similar"] = "similar",
    *,
    seed: int = 0,
) -> dict[str, list[str]]:
    """``k`` partners for every subject.

    ``similar`` takes the subjects with the closest feature (``|log f_a - log f_b|``, e.g. a
    volume); subjects without a positive feature fall back to ``random``.
    """
    ids = sorted(features)
    g = torch.Generator().manual_seed(seed)
    out: dict[str, list[str]] = {}
    for a in ids:
        others = [b for b in ids if b != a]
        fa = features[a]
        if method == "similar" and fa is not None and fa > 0:
            scored = [(abs(math.log(fa / fb)), b) for b in others if (fb := features[b]) and fb > 0]
            out[a] = [b for _, b in sorted(scored)[:k]]
        else:
            perm = torch.randperm(len(others), generator=g).tolist()
            out[a] = [others[i] for i in perm[:k]]
    return out


class FieldBank:
    """A directory of pairwise SVFs. Opens ``root`` if it holds a bank, else starts a new one.

    ``profile``: the site profile used for masks, registration, partner selection, storage and QC;
    stored in the bank. Required for a new bank unless the defaults are wanted.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        profile: SiteProfile | None = None,
        cache_bytes: int = 2 * 2**30,
    ) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[Any, ...], Tensor] = OrderedDict()
        self.cache_bytes = cache_bytes
        self.profile: SiteProfile = profile or SiteProfile(name="default")
        self.subjects: dict[str, Subject] = {}
        self.pairs: dict[tuple[str, str], Pair] = {}
        index = self.root / "bank.json"
        if index.exists():
            self._load(index)
            if profile is not None and profile != self.profile:
                raise ValueError(f"{root} was built with a different profile ({self.profile.name})")

    # --- persistence ---------------------------------------------------------------------------

    def _load(self, index: Path) -> None:
        data = json.loads(index.read_text())
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported bank format {data.get('format')!r}")
        self.profile = SiteProfile.from_dict(data["profile"])
        for sid, s in data["subjects"].items():
            self.subjects[sid] = Subject(
                sid, s["image"], s.get("labels"), _space_from_json(s["space"]), s.get("feature")
            )
        for p in data["pairs"]:
            self.pairs[(p["a"], p["b"])] = Pair(**p)

    def save(self) -> None:
        with self._lock:
            data = {
                "format": FORMAT_VERSION,
                "profile": self.profile.to_dict(),
                "subjects": {
                    s.id: {
                        "image": s.image,
                        "labels": s.labels,
                        "space": _space_to_json(s.space),
                        "feature": s.feature,
                    }
                    for s in self.subjects.values()
                },
                "pairs": [p.__dict__ for p in self.pairs.values()],
            }
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "bank.json.tmp"
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.root / "bank.json")

    def _rel(self, path: PathLike) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    def _abs(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    # --- subjects ------------------------------------------------------------------------------

    def add_subject(self, sid: str, image: PathLike, labels: PathLike | None = None) -> Subject:
        """Register a (preprocessed) NIfTI image, and optionally its label map, under ``sid``."""
        if not _ID.match(sid):
            raise ValueError(f"subject id {sid!r} must match {_ID.pattern}")
        img, space = load_nifti(image)
        lab = None
        if labels is not None:
            lab, lspace = load_nifti(labels, dtype=torch.int64)
            if lspace.shape != space.shape or not np.allclose(lspace.affine, space.affine, 1e-4):
                raise ValueError(f"labels of {sid} are not on the image grid")
        mask = self._registration_mask(img, lab, space)
        pcfg = self.profile.partners
        if pcfg.feature_source == "foreground":
            feature_src: Tensor | None = foreground_mask(img, space)
        elif lab is not None:
            feature_src = label_mask(lab, pcfg.feature_labels)
        else:
            feature_src = mask
        feature = None
        if feature_src is not None:
            feature = float(feature_src.sum()) * float(abs(np.linalg.det(space.affine)))
        s = Subject(sid, self._rel(image), None if labels is None else self._rel(labels), space)
        s.feature = feature
        with self._lock:
            self.subjects[sid] = s
        if mask is not None:
            (self.root / "masks").mkdir(parents=True, exist_ok=True)
            save_nifti(self.mask_path(sid), mask.to(torch.uint8), space)
        self.save()
        return s

    def _registration_mask(
        self, image: Tensor, labels: Tensor | None, space: ImageSpace
    ) -> Tensor | None:
        cfg = self.profile.registration.mask
        if cfg is None:
            return None
        if labels is not None and cfg.labels:
            m = label_mask(labels, cfg.labels)
        elif cfg.threshold is not None:
            m = foreground_mask(
                image, space, cfg.threshold, smooth_mm=cfg.smooth_mm, open_mm=cfg.open_mm
            )
        else:
            return None
        return dilate(m, space, cfg.dilate_mm)

    def mask_path(self, sid: str) -> Path:
        return self.root / "masks" / f"{sid}.nii.gz"

    def _mask(self, sid: str) -> Path | None:
        p = self.mask_path(sid)
        return p if p.exists() else None

    # --- building ------------------------------------------------------------------------------

    def plan_pairs(
        self,
        k: int | None = None,
        method: Literal["random", "similar"] | None = None,
        seed: int = 0,
    ) -> list[tuple[str, str]]:
        """Partner pairs ``(A, B)`` from the profile's partner settings (overridable)."""
        cfg = self.profile.partners
        chosen = select_partners(
            {s.id: s.feature for s in self.subjects.values()},
            cfg.k if k is None else k,
            cfg.method if method is None else method,
            seed=seed,
        )
        return [(a, b) for a, bs in chosen.items() for b in bs]

    def default_registration(self, threads: int | None = None) -> Registration:
        """The profile's backend (``threads``: OpenMP threads for NiftyReg)."""
        return self.profile.registration.make(threads)

    def build(
        self,
        pairs: Iterable[tuple[str, str]] | None = None,
        *,
        registration: Registration | None = None,
        jobs: int = 1,
        overwrite: bool = False,
        kind: Literal["inter", "intra"] = "inter",
        callback: Callable[[Pair], None] | None = None,
    ) -> list[Pair]:
        """Register every pair (default :meth:`plan_pairs`) and store the fields.

        Pairs already in the bank are skipped unless ``overwrite``, so an interrupted build can be
        resumed. ``jobs`` pairs run in parallel threads; the NiftyReg backend then gets
        ``cpu_count // jobs`` OpenMP threads each. With the torch backend on one GPU, keep
        ``jobs=1``. A failing registration is logged and skipped.
        """
        todo = [p for p in (self.plan_pairs() if pairs is None else pairs)]
        todo = [p for p in todo if overwrite or p not in self.pairs]
        if registration is None:
            registration = self.default_registration(max(1, (os.cpu_count() or 1) // jobs))
        reg = registration
        if jobs > 1 and not getattr(reg, "thread_safe", True):
            log.warning("%s is not thread-safe; registering sequentially", type(reg).__name__)
            jobs = 1

        def one(ab: tuple[str, str]) -> Pair | None:
            try:
                pair = self.register_pair(*ab, registration=reg, kind=kind)
            except Exception:
                log.exception("registration of %s <- %s failed", *ab)
                return None
            if callback is not None:
                callback(pair)
            return pair

        if jobs <= 1:
            results = [one(ab) for ab in todo]
        else:
            with ThreadPoolExecutor(jobs) as ex:
                results = list(ex.map(one, todo))
        return [p for p in results if p is not None]

    def register_pair(
        self,
        a: str,
        b: str,
        *,
        registration: Registration | None = None,
        kind: Literal["inter", "intra"] = "inter",
    ) -> Pair:
        """Register B (floating) to A (reference) and store the result.

        With ``profile.registration.spacing`` set, both images (and masks) are first resampled to
        that spacing; the velocity is then re-expressed on A's full grid. The fields are smooth
        (control points every few mm), so this loses little and saves a lot of time and memory.

        With ``profile.registration.compartments`` set and labels for both subjects, each
        compartment is also registered on its own (see :meth:`_register_compartments`).
        """
        reg = registration or self.default_registration()
        result = self._register(reg, a, b, self._mask(a), self._mask(b))
        cfg = self.profile.registration
        if cfg.compartments and self.subjects[a].labels and self.subjects[b].labels:
            result = self._register_compartments(reg, a, b, result)
        return self.add_result(a, b, result, kind=kind)

    def _register(
        self,
        reg: Registration,
        a: str,
        b: str,
        rmask: Path | None,
        fmask: Path | None,
        affine: np.ndarray | None = None,
    ) -> RegistrationResult:
        """One registration, at ``registration.spacing`` if set; result on A's full grid."""
        sa = self.subjects[a]
        spacing = self.profile.registration.spacing
        images = [self._abs(sa.image), self._abs(self.subjects[b].image)]
        if spacing is None:
            return reg.register(*images, reference_mask=rmask, floating_mask=fmask, affine=affine)
        with tempfile.TemporaryDirectory(prefix="drift_bank_") as tmp:

            def shrink(path: Path, name: str, mode: Literal["linear", "nearest"]) -> Path:
                img, space = load_nifti(path)
                out = Path(tmp) / f"{name}.nii.gz"
                small, small_space = resample(img, space, spacing, mode=mode)
                save_nifti(out, small, small_space)
                return out

            ref, flo = (shrink(p, f"image{i}", "linear") for i, p in enumerate(images))
            rm, fm = (
                None if p is None else shrink(p, f"mask{i}", "nearest")
                for i, p in enumerate((rmask, fmask))
            )
            coarse = reg.register(ref, flo, reference_mask=rm, floating_mask=fm, affine=affine)
        v = resample_field(coarse.velocity.float(), coarse.space, sa.space)
        return RegistrationResult(
            space=sa.space,
            affine=coarse.affine,
            velocity=v,
            jacobian=jacobian_stats(exp(v)),
            qc={**coarse.qc, "registration_spacing_mm": float(spacing)},
        )

    def _register_compartments(
        self, reg: Registration, a: str, b: str, base: RegistrationResult
    ) -> RegistrationResult:
        """Register each compartment separately and blend the velocities (sliding interfaces).

        Organs that slide along each other (e.g. lungs along the chest wall) can't be matched by
        one smooth deformation. Each compartment (a label set, or "the rest of the body" when its
        label list is empty) is registered with its own masks, relative to the whole-body affine
        of ``base``. The velocities are blended with smoothed compartment indicators of A,
        ``v = (Σ s_k v_k + r v_base) / (Σ s_k + r)`` with ``r = max(0, 1 - Σ s_k)`` (``v_base``
        outside all compartments), and exponentiated once, so the result is still a single
        diffeomorphism. True sliding is discontinuous; here it is approximated by a transition
        zone of width ``blend_sigma_mm``.
        """
        cfg = self.profile.registration
        sa, sb = self.subjects[a], self.subjects[b]
        la, _ = load_nifti(self._abs(str(sa.labels)), dtype=torch.int64)
        lb, _ = load_nifti(self._abs(str(sb.labels)), dtype=torch.int64)
        listed = [lab for c in cfg.compartments.values() for lab in c.labels]

        def body(sid: str, labels: Tensor) -> Tensor:
            m = self._mask(sid)
            return (load_nifti(m)[0] > 0) if m is not None else labels >= 0

        sigma = [cfg.blend_sigma_mm / s for s in sa.space.spacing]
        num = torch.zeros_like(base.velocity, dtype=torch.float32)
        total = torch.zeros(1, 1, *sa.space.shape)
        qc: dict[str, float] = {}
        with tempfile.TemporaryDirectory(prefix="drift_compartments_") as tmp:
            for name, comp in cfg.compartments.items():
                masks = []
                for sid, labels, space in ((a, la, sa.space), (b, lb, sb.space)):
                    if comp.labels:
                        m = label_mask(labels, comp.labels)
                    else:  # the rest of the body
                        m = body(sid, labels) & ~label_mask(labels, listed)
                    masks.append((m, space))
                if not (bool(masks[0][0].any()) and bool(masks[1][0].any())):
                    log.info("compartment %s missing in %s or %s; skipped", name, a, b)
                    continue
                paths = []
                for i, (m, space) in enumerate(masks):
                    path = Path(tmp) / f"{name}_{i}.nii.gz"
                    save_nifti(path, dilate(m, space, comp.dilate_mm).to(torch.uint8), space)
                    paths.append(path)
                r = self._register(reg, a, b, paths[0], paths[1], affine=base.affine)
                weight = gaussian_smooth(masks[0][0].float(), sigma)
                num = num + weight * r.velocity.float()
                total = total + weight
                qc[f"compartment_{name}_det_min"] = float(r.jacobian.min[0])
        if not qc:
            return base
        rest = (1.0 - total).clamp_min(0.0)
        v = (num + rest * base.velocity.float()) / (total + rest).clamp_min(1e-6)
        return RegistrationResult(
            space=sa.space,
            affine=base.affine,
            velocity=v,
            jacobian=jacobian_stats(exp(v)),
            qc={**base.qc, **qc, "compartments": float(len(qc))},
        )

    def add_result(
        self,
        a: str,
        b: str,
        result: RegistrationResult,
        *,
        kind: Literal["inter", "intra"] = "inter",
    ) -> Pair:
        """Store a registration of B to A, compute its QC and decide whether it passes."""
        sa = self.subjects[a]
        if result.space.shape != sa.space.shape or not np.allclose(
            result.space.affine, sa.space.affine, atol=1e-4
        ):
            raise ValueError(f"registration result is not on {a}'s grid")
        v = result.velocity.detach().float().cpu()
        js = result.jacobian
        qc: dict[str, float] = {
            "det_min": float(js.min[0]),
            "det_max": float(js.max[0]),
            "det_p01": float(js.p01[0]),
            "det_p99": float(js.p99[0]),
            "fold_fraction": float(js.fold_fraction[0]),
            "max_disp_mm": float(sa.space.displacement_to_world(v).norm(dim=1).max()),
            **{k: float(x) for k, x in result.qc.items()},
        }
        qc.update(self._dice(a, b, v, result.affine))

        store = sa.space
        if self.profile.store_spacing is not None:
            store = with_spacing(sa.space, self.profile.store_spacing)
        stored = resample_field(v, sa.space, store) if store is not sa.space else v
        rel = Path("fields") / a / f"{b}.npz"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            velocity=stored[0].numpy().astype(np.float16),
            shape=np.asarray(store.shape),
            affine=np.asarray(store.affine),
            registration_affine=np.asarray(result.affine),
        )
        passed, reasons = self._check(qc)
        pair = Pair(a, b, kind, str(rel), qc, passed, reasons)
        with self._lock:
            self.pairs[(a, b)] = pair
            for key in [k for k in self._cache if k[:2] == (a, b)]:
                del self._cache[key]
        self.save()
        return pair

    def _dice(self, a: str, b: str, v: Tensor, affine: np.ndarray) -> dict[str, float]:
        sa, sb = self.subjects[a], self.subjects[b]
        if sa.labels is None or sb.labels is None:
            return {}
        sel = self.profile.qc.pair_dice_labels
        la, _ = load_nifti(self._abs(sa.labels), dtype=torch.int64)
        lb, _ = load_nifti(self._abs(sb.labels), dtype=torch.int64)
        ma = label_mask(la, sel).long()
        mb = resample_to(
            label_mask(lb, sel).long(), sb.space, sa.space, affine=affine, mode="nearest"
        )
        u = exp(-v)
        warped = warp_labels(ma, u, num_classes=2)
        out = {"dice": dice(warped, mb), "dice_affine": dice(ma, mb)}
        if self.profile.qc.pair_dice_each and sel:
            lb_on_a = resample_to(lb, sb.space, sa.space, affine=affine, mode="nearest")
            la_warped = warp_labels(la, u, num_classes=max(int(la.max()), *sel) + 1)
            for lab in sel:
                if bool((la == lab).any()) and bool((lb_on_a == lab).any()):
                    out[f"dice_{lab}"] = dice(la_warped == lab, lb_on_a == lab)
                    out[f"dice_affine_{lab}"] = dice(la == lab, lb_on_a == lab)
        return out

    def _check(self, qc: Mapping[str, float]) -> tuple[bool, list[str]]:
        c = self.profile.qc
        reasons = []
        if qc["det_min"] < c.pair_det_min:
            reasons.append(f"det_min {qc['det_min']:.3g} < {c.pair_det_min}")
        if qc["det_max"] > c.pair_det_max:
            reasons.append(f"det_max {qc['det_max']:.3g} > {c.pair_det_max}")
        if qc["fold_fraction"] > c.pair_max_fold_fraction:
            reasons.append(f"fold_fraction {qc['fold_fraction']:.3g} > {c.pair_max_fold_fraction}")
        if c.pair_min_dice is not None and "dice" in qc and qc["dice"] < c.pair_min_dice:
            reasons.append(f"dice {qc['dice']:.3g} < {c.pair_min_dice}")
        return not reasons, reasons

    def recheck(self) -> None:
        """Re-apply the profile's pair QC bounds to every stored pair (e.g. after editing them)."""
        with self._lock:
            for p in self.pairs.values():
                p.passed, p.reasons = self._check(p.qc)
        self.save()

    # --- using ---------------------------------------------------------------------------------

    def partners(self, a: str, *, passed_only: bool = True) -> list[str]:
        """Partners of ``a`` with a stored field (only those that passed QC by default)."""
        return sorted(
            p.b for p in self.pairs.values() if p.a == a and (p.passed or not passed_only)
        )

    def velocity(
        self,
        a: str,
        b: str,
        space: ImageSpace | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Augmentation velocity ``(1, D, *space.shape)`` that moves A towards B's shape.

        ``warp(A, exp(vel, t))`` is A at ``t = 0`` and has B's (non-rigid) shape at ``t = 1``;
        other ``t`` interpolate or extrapolate, and a weight map gives regional times. ``space``
        is the grid of A's image as it will be augmented (default: the grid A was registered on);
        it must describe the same world frame. The result is cached and shared: don't modify it in
        place.
        """
        pair = self.pairs[(a, b)]
        space = space or self.subjects[a].space
        dev = torch.device(device) if device is not None else torch.device("cpu")
        key = (a, b, space.shape, np.asarray(space.affine).tobytes(), str(dev))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with np.load(self._abs(pair.field)) as f:
            stored = torch.from_numpy(f["velocity"].astype(np.float32))[None].to(dev)
            store = ImageSpace(tuple(int(n) for n in f["shape"]), f["affine"])
        if store.shape == space.shape and np.allclose(store.affine, space.affine, atol=1e-6):
            vel = -stored
        else:
            vel = -resample_field(stored, store, space)
        with self._lock:
            self._cache[key] = vel
            # bounded by memory, not count: a full-resolution 3D field is ~200 MB
            while len(self._cache) > 1 and self._cached_bytes() > self.cache_bytes:
                self._cache.popitem(last=False)
        return vel

    def _cached_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._cache.values())

    def summary(self) -> dict[str, int]:
        passed = sum(p.passed for p in self.pairs.values())
        return {
            "subjects": len(self.subjects),
            "pairs": len(self.pairs),
            "passed": passed,
            "failed": len(self.pairs) - passed,
        }
