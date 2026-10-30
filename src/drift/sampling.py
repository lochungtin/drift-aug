"""Random time policies and per-sample rejection rules for the augmentation.

A policy draws the times for one augmentation: a single ``t`` (:class:`GlobalPolicy`), a weight map
from the subject's labels (:class:`RegionalPolicy`), random blobs (:class:`BlobPolicy`) or a random
choice between policies (:class:`MixPolicy`). All randomness comes from the ``torch.Generator``
passed in, so a seeded generator gives reproducible draws. Rules (:class:`JacobianBounds`,
:class:`MaxDisplacement`) reject a sampled displacement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from drift import weights
from drift.jacobian import det
from drift.preprocess import label_mask
from drift.profile import QCConfig, SamplingConfig
from drift.space import ImageSpace


class Distribution(Protocol):
    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``n`` draws as a float64 CPU tensor."""
        ...


@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        u = torch.rand(n, generator=generator, dtype=torch.float64)
        return self.lo + (self.hi - self.lo) * u


@dataclass(frozen=True)
class Constant:
    value: float

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        return torch.full((n,), float(self.value), dtype=torch.float64)


def _draw(dist: Distribution, generator: torch.Generator | None) -> float:
    return float(dist.sample(1, generator)[0])


@dataclass(frozen=True)
class Draw:
    """One policy draw. ``w`` is a ``(1,)`` time or a ``(1, 1, *spatial)`` weight map; ``times``
    records what was drawn (for logging)."""

    w: Tensor
    times: dict[str, float]

    @property
    def regional(self) -> bool:
        return self.w.ndim > 1


class Policy(Protocol):
    def sample(
        self, labels: Tensor | None, space: ImageSpace, generator: torch.Generator | None = None
    ) -> Draw:
        """Draw times for one sample. ``labels``: ``(1, 1, *space.shape)`` integer map or None."""
        ...


@dataclass(frozen=True)
class GlobalPolicy:
    """One ``t`` for the whole image."""

    t: Distribution = field(default_factory=lambda: Uniform(-0.25, 1.25))

    def sample(
        self, labels: Tensor | None, space: ImageSpace, generator: torch.Generator | None = None
    ) -> Draw:
        t = _draw(self.t, generator)
        device = labels.device if labels is not None else None
        return Draw(torch.tensor([t], device=device), {"t": t})


@dataclass(frozen=True)
class Region:
    """Labels sharing one ``t``; see :class:`drift.profile.RegionConfig`."""

    labels: tuple[int, ...]
    t: Distribution | None = None
    follow_background: bool = False
    split_axis: int | None = None


@dataclass(frozen=True)
class RegionalPolicy:
    """Per-region times from the labels, on a background time ``t_bg``, smoothed by ``sigma_mm``.

    Regions are painted in order (later regions on top of earlier ones); regions absent from a
    subject's labels are skipped. Without labels this falls back to a global ``t``.
    """

    regions: Mapping[str, Region]
    t: Distribution = field(default_factory=lambda: Uniform(-0.25, 1.25))
    t_bg: Distribution = field(default_factory=lambda: Uniform(0.0, 0.5))
    sigma_mm: float = 6.0

    def sample(
        self, labels: Tensor | None, space: ImageSpace, generator: torch.Generator | None = None
    ) -> Draw:
        if labels is None:
            return GlobalPolicy(self.t).sample(None, space, generator)
        t_bg = _draw(self.t_bg, generator)
        ids = torch.zeros_like(labels, dtype=torch.int64)
        times: dict[str, float] = {"background": t_bg}
        id_times: dict[int, float] = {}
        for name, region in self.regions.items():
            mask = label_mask(labels, region.labels)
            if not bool(mask.any()):
                continue
            parts = [(name, mask)]
            if region.split_axis is not None:
                ax = region.split_axis
                idx = torch.arange(labels.shape[2 + ax], device=labels.device)
                idx = idx.view(*([1] * (2 + ax)), -1, *([1] * (labels.ndim - 3 - ax)))
                centre = (idx * mask).sum() / mask.sum()
                upper = (idx >= centre).expand_as(mask)
                parts = [(f"{name}[0]", mask & ~upper), (f"{name}[1]", mask & upper)]
            for part, m in parts:
                t = t_bg if region.follow_background else _draw(region.t or self.t, generator)
                k = len(id_times) + 1
                ids[m] = k
                id_times[k] = t
                times[part] = t
        w = weights.from_labels(
            ids,
            id_times,
            t_bg=t_bg,
            sigma=weights.sigma_voxels(self.sigma_mm, space),
            dtype=torch.float32,
        )
        return Draw(w, times)


@dataclass(frozen=True)
class BlobPolicy:
    """``n`` random Gaussian blobs with times from ``U(t)`` on a background ``t_bg``.

    Blob centres are drawn inside ``within_labels`` when given (and present in the labels).
    ``sigma_mm`` is converted with the mean voxel spacing.
    """

    n: int = 3
    t: Uniform = field(default_factory=lambda: Uniform(-0.25, 1.25))
    t_bg: Distribution = field(default_factory=lambda: Uniform(0.0, 0.5))
    sigma_mm: tuple[float, float] = (10.0, 25.0)
    within_labels: tuple[int, ...] | None = None

    def sample(
        self, labels: Tensor | None, space: ImageSpace, generator: torch.Generator | None = None
    ) -> Draw:
        t_bg = _draw(self.t_bg, generator)
        within = None
        if labels is not None and self.within_labels is not None:
            within = label_mask(labels, self.within_labels)
            if not bool(within.any()):
                within = None
        mm = float(space.spacing.mean())
        w = weights.random_blobs(
            space.shape,
            self.n,
            t=(self.t.lo, self.t.hi),
            sigma=(self.sigma_mm[0] / mm, self.sigma_mm[1] / mm),
            t_bg=t_bg,
            within=within,
            generator=generator,
            dtype=torch.float32,
            device=labels.device if labels is not None else None,
        )
        return Draw(w, {"background": t_bg})


@dataclass(frozen=True)
class MixPolicy:
    """Pick one of ``policies`` at random with probabilities ``probs`` (uniform if None)."""

    policies: Sequence[Policy]
    probs: Sequence[float] | None = None

    def sample(
        self, labels: Tensor | None, space: ImageSpace, generator: torch.Generator | None = None
    ) -> Draw:
        p = torch.as_tensor(
            self.probs if self.probs is not None else [1.0] * len(self.policies),
            dtype=torch.float64,
        )
        i = int(torch.multinomial(p, 1, generator=generator))
        return self.policies[i].sample(labels, space, generator)


def policy_from_config(cfg: SamplingConfig) -> Policy:
    """The policy described by a profile's ``sampling`` section."""
    glob = GlobalPolicy(Uniform(*cfg.t))
    if cfg.policy == "global":
        return glob
    regions = {
        name: Region(
            labels=r.labels,
            t=Uniform(*r.t) if r.t is not None else None,
            follow_background=r.follow_background,
            split_axis=r.split_axis,
        )
        for name, r in cfg.regions.items()
    }
    regional = RegionalPolicy(regions, Uniform(*cfg.t), Uniform(*cfg.t_bg), cfg.sigma_mm)
    if cfg.policy == "regional":
        return regional
    return MixPolicy([regional, glob], [cfg.p_regional, 1.0 - cfg.p_regional])


# --- rejection rules ---------------------------------------------------------------------------


class Rule(Protocol):
    def check(self, disp: Tensor, space: ImageSpace) -> Tensor:
        """Per-sample ``(B,)`` bool: True if the displacement ``(B, D, *spatial)`` is accepted."""
        ...


@dataclass(frozen=True)
class JacobianBounds:
    """Accept if ``lo <= det J <= hi`` everywhere."""

    lo: float = 0.3
    hi: float = 3.0

    def check(self, disp: Tensor, space: ImageSpace) -> Tensor:
        dj = det(disp).flatten(1)
        return (dj.amin(1) >= self.lo) & (dj.amax(1) <= self.hi)


@dataclass(frozen=True)
class MaxDisplacement:
    """Accept if no voxel moves more than ``mm`` millimetres."""

    mm: float

    def check(self, disp: Tensor, space: ImageSpace) -> Tensor:
        mag: Tensor = space.displacement_to_world(disp).norm(dim=1).flatten(1).amax(1)
        return mag <= self.mm


def rules_from_config(cfg: QCConfig) -> list[Rule]:
    """Per-sample rules from a profile's ``qc`` section."""
    rules: list[Rule] = [JacobianBounds(cfg.det_min, cfg.det_max)]
    if cfg.max_disp_mm is not None:
        rules.append(MaxDisplacement(cfg.max_disp_mm))
    return rules
