"""``DriftAugment``: move a subject partway along a stored inter-patient deformation.

Each call picks a partner from the :class:`~drift.bank.FieldBank`, draws times from a policy
(:mod:`drift.sampling`), builds ``φ`` (Lagrangian :func:`~drift.flow.integrate` for weight maps by
default, else :func:`~drift.flow.exp`), rejects it if it breaks a rule (retrying with a new
partner and new times, then falling back to the identity), and warps every image and label key
with the same ``φ``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from drift.bank import FieldBank
from drift.flow import exp, integrate
from drift.grid import identity, warp, warp_labels
from drift.preprocess import resample_field, resample_to, with_spacing
from drift.sampling import Draw, Policy, Rule, policy_from_config, rules_from_config
from drift.space import ImageSpace
from drift.weights import MAX_RATE, limit_rate


class DriftAugment:
    """Dictionary transform for segmentation training.

    The input dict holds the subject id (``subject_key``), images (``image_keys``, linear
    interpolation, zero padding) and label maps (``label_keys``), each ``(C, *spatial)`` or
    ``(1, C, *spatial)``, as torch tensors or NumPy arrays. They must lie on the grid the subject
    was registered on, unless the dict carries the grid as an :class:`ImageSpace` under
    ``space_key``, in which case the field is resampled onto it (same world frame, any spacing).
    Regional policies use the first label key. A partner id under ``partner_key`` fixes the
    partner (its field is used even if it failed pair QC); otherwise one is drawn among the
    subject's partners that passed.

    Outputs replace the inputs (same types and shapes). Extra keys: ``"drift"`` (partner, times,
    attempts, whether a sample was accepted), ``valid_mask_key`` (voxels whose source lies inside
    the image) and ``"drift_displacement"`` (``(D, *spatial)``, voxels) if ``return_field``.

    ``field_spacing`` (mm): compute the deformation on a grid of that spacing and upsample it to
    the image grid. Fields from the bank are smooth (control points several mm apart), so this
    changes samples very little and makes regional (Lagrangian) flows several times cheaper.

    Defaults for ``policy``, ``lagrangian`` and ``rules`` come from the bank's site profile.
    ``seed`` makes the draws reproducible; in DataLoader workers each worker gets its own stream.
    """

    def __init__(
        self,
        bank: FieldBank,
        policy: Policy | None = None,
        *,
        lagrangian: bool | None = None,
        rules: Sequence[Rule] | None = None,
        max_retries: int = 5,
        max_rate: float = MAX_RATE,
        image_keys: Sequence[str] = ("image",),
        label_keys: Sequence[str] = ("label",),
        subject_key: str = "subject",
        space_key: str = "space",
        partner_key: str = "partner",
        label_mode: Literal["soft", "nearest"] = "soft",
        valid_mask_key: str | None = None,
        return_field: bool = False,
        prob: float = 1.0,
        field_spacing: float | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.bank = bank
        profile = bank.profile
        self.policy = policy if policy is not None else policy_from_config(profile.sampling)
        self.lagrangian = profile.sampling.lagrangian if lagrangian is None else lagrangian
        self.rules = list(rules) if rules is not None else rules_from_config(profile.qc)
        self.max_retries = max_retries
        self.max_rate = max_rate
        self.image_keys = tuple(image_keys)
        self.label_keys = tuple(label_keys)
        self.subject_key = subject_key
        self.space_key = space_key
        self.partner_key = partner_key
        self.label_mode = label_mode
        self.valid_mask_key = valid_mask_key
        self.return_field = return_field
        self.prob = prob
        self.field_spacing = field_spacing
        self.seed = seed
        self.device = torch.device(device) if device is not None else None
        self._gen: torch.Generator | None = None
        self._gen_owner: tuple[int, int] | None = None

    # --- randomness ----------------------------------------------------------------------------

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self._gen = None

    @property
    def generator(self) -> torch.Generator:
        info = torch.utils.data.get_worker_info()
        owner = (os.getpid(), -1 if info is None else info.id)
        if self._gen is None or self._gen_owner != owner:
            if self.seed is None:
                base = torch.initial_seed()  # differs per DataLoader worker
            else:
                base = self.seed + (0 if info is None else 1_000_003 * (info.id + 1))
            self._gen = torch.Generator().manual_seed(base % 2**63)
            self._gen_owner = owner
        return self._gen

    def _randint(self, n: int) -> int:
        return int(torch.randint(n, (1,), generator=self.generator))

    # --- the transform -------------------------------------------------------------------------

    def sample_displacement(
        self,
        subject: str,
        space: ImageSpace,
        labels: Tensor | None,
        device: torch.device,
        partner: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Draw one accepted displacement ``(1, D, *space.shape)`` for ``subject``."""
        if partner is not None and (subject, partner) not in self.bank.pairs:
            raise KeyError(f"no field {subject} <- {partner} in the bank")
        if self.field_spacing is None:
            return self._sample(subject, space, labels, device, partner)
        # Compute the (smooth) deformation on a coarser grid of the same world frame, then
        # re-express it on the image grid: far cheaper for Lagrangian regional flows.
        coarse = with_spacing(space, self.field_spacing)
        coarse_labels = None
        if labels is not None:
            coarse_labels = resample_to(labels, space, coarse, mode="nearest")
        disp, meta = self._sample(subject, coarse, coarse_labels, device, partner)
        return resample_field(disp, coarse, space), meta

    def _sample(
        self,
        subject: str,
        space: ImageSpace,
        labels: Tensor | None,
        device: torch.device,
        partner: str | None,
    ) -> tuple[Tensor, dict[str, Any]]:
        partners = [partner] if partner is not None else self.bank.partners(subject)
        zero = torch.zeros(1, space.ndim, *space.shape, device=device)
        if not partners:
            return zero, {"partner": None, "accepted": False, "attempts": 0}
        meta: dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 1):
            chosen = partners[self._randint(len(partners))]
            v = self.bank.velocity(subject, chosen, space, device=device)
            draw = self.policy.sample(labels, space, self.generator)
            disp = self._flow(v, draw)
            meta = {
                "partner": chosen,
                "times": draw.times,
                "regional": draw.regional,
                "attempts": attempt,
            }
            if all(bool(r.check(disp, space).all()) for r in self.rules):
                return disp, {**meta, "accepted": True}
        return zero, {**meta, "accepted": False}

    def _flow(self, v: Tensor, draw: Draw) -> Tensor:
        w = draw.w.to(v)
        if not draw.regional:
            return exp(v, float(w[0]))
        if self.lagrangian:
            return integrate(v, limit_rate(w, v, self.max_rate))
        return exp(v, w)

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        subject = str(sample[self.subject_key])
        info = self.bank.subjects[subject]
        d = info.space.ndim
        keys = [k for k in (*self.image_keys, *self.label_keys) if k in sample]
        if not keys:
            raise KeyError(f"none of {self.image_keys + self.label_keys} in the sample")
        tensors = {k: _to_batched(sample[k], d) for k in keys}
        shape = tuple(next(iter(tensors.values()))[0].shape[2:])
        space = sample.get(self.space_key)
        if not isinstance(space, ImageSpace):
            space = info.space
        if tuple(space.shape) != shape:
            raise ValueError(
                f"{subject}: image shape {shape} does not match its grid {space.shape}; pass the "
                f"image's ImageSpace under {self.space_key!r}"
            )
        device = self.device or tensors[keys[0]][0].device

        if self.prob < 1.0 and float(torch.rand(1, generator=self.generator)) >= self.prob:
            disp = torch.zeros(1, d, *shape, device=device)
            meta: dict[str, Any] = {
                "partner": None,
                "accepted": False,
                "attempts": 0,
                "skipped": True,
            }
        else:
            label_key = next((k for k in self.label_keys if k in tensors), None)
            labels = None
            if label_key is not None:
                labels = tensors[label_key][0].to(device=device, dtype=torch.int64)[:, :1]
            partner = sample.get(self.partner_key)
            disp, meta = self.sample_displacement(
                subject, space, labels, device, None if partner is None else str(partner)
            )

        for k in keys if meta["accepted"] else ():  # the identity leaves inputs untouched
            x, restore = tensors[k]
            x = x.to(device)
            if k in self.label_keys:
                y = warp_labels(x.long(), disp, mode=self.label_mode).to(x.dtype)
            else:
                y = warp(x.to(disp.dtype), disp, padding_mode="zeros")
                y = y.to(x.dtype) if x.is_floating_point() else y.round().to(x.dtype)
            out[k] = restore(y)
        if self.valid_mask_key is not None:
            coords = identity(shape, device=device, dtype=disp.dtype) + disp
            hi = torch.tensor([n - 1 for n in shape], device=device, dtype=disp.dtype)
            inside = ((coords >= 0) & (coords <= hi.view(1, d, *[1] * d))).all(1, keepdim=True)
            out[self.valid_mask_key] = _restore_like(sample[keys[0]], inside, d)
        if self.return_field:
            out["drift_displacement"] = disp[0]
        out["drift"] = meta
        return out


def _to_batched(x: Any, d: int) -> tuple[Tensor, Any]:
    """``(1, C, *spatial)`` tensor plus a function that converts a result back to ``x``'s form."""
    t = torch.from_numpy(np.asarray(x)) if not isinstance(x, Tensor) else x
    if t.ndim == d + 1:
        t = t[None]
    elif t.ndim != d + 2 or t.shape[0] != 1:
        raise ValueError(f"expected (C, *spatial) or (1, C, *spatial) with {d} spatial dims")

    def restore(y: Tensor) -> Any:
        return _restore_like(x, y, d)

    return t, restore


def _restore_like(x: Any, y: Tensor, d: int) -> Any:
    ndim = x.ndim if hasattr(x, "ndim") else np.asarray(x).ndim
    if ndim == d + 1:
        y = y[0]
    if isinstance(x, Tensor):
        return y.to(x.device)
    return y.cpu().numpy()
