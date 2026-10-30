"""Weight maps ``w(x)`` for regional flows ``exp(w(x)·v)``.

A weight map has shape ``(B, 1, *spatial)`` and gives each voxel its own time ``t``. Values may lie
outside ``[0, 1]`` (extrapolation). Maps are built from labels (:func:`from_labels`), random
blobs (:func:`random_blobs`) or a constant (:func:`constant`), and must be smooth: a sharp edge
concentrates all the strain in a thin ring, and in Lagrangian mode (:func:`drift.flow.integrate`)
it makes tissue collide. :func:`rate` measures this and :func:`limit_rate` enforces it.

Anywhere a time is accepted, it may be a float or a per-sample ``(B,)`` tensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from drift.filters import gaussian_smooth
from drift.grid import _check_field, identity
from drift.space import ImageSpace

Time = float | Tensor
Sigma = float | Sequence[float]

#: Default bound on ``max |∇w · v|``. The Lagrangian map stops being invertible at 1.
MAX_RATE = 0.5


def sigma_voxels(sigma_mm: float, space: ImageSpace) -> tuple[float, ...]:
    """Convert an isotropic Gaussian ``sigma`` in mm to per-axis voxels for ``space``."""
    return tuple(float(sigma_mm / s) for s in space.spacing)


def _per_sample(t: Time, ndim: int, *, dtype: torch.dtype, device: torch.device | None) -> Tensor:
    """``t`` as a tensor of shape ``(B or 1, 1, *[1]*ndim)``."""
    tt = torch.as_tensor(t, dtype=dtype, device=device)
    if tt.ndim > 1:
        raise ValueError(f"a time must be a float or a (B,) tensor, got shape {tuple(tt.shape)}")
    return tt.reshape(-1, *([1] * (ndim + 1)))


def constant(
    t: Time,
    shape: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Spatially constant map ``(B, 1, *shape)``; ``B`` is 1 for a float ``t``."""
    dtype = dtype or torch.get_default_dtype()
    tt = _per_sample(t, len(shape), dtype=dtype, device=device)
    return tt.expand(tt.shape[0], 1, *shape).clone()


def from_labels(
    labels: Tensor,
    t: Mapping[int, Time],
    *,
    t_bg: Time = 0.0,
    sigma: Sigma = 0.0,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Per-structure times from an integer label map ``(B, 1, *spatial)``.

    Voxels with label ``k`` get ``t[k]``; every other voxel (unlisted labels included) gets
    ``t_bg``. The piecewise-constant map is then smoothed with a Gaussian of ``sigma`` voxels (per
    axis allowed; see :func:`sigma_voxels` for mm), so values stay within the range of the times.
    """
    if labels.ndim < 3 or labels.shape[1] != 1:
        raise ValueError(f"labels must be (B, 1, *spatial), got shape {tuple(labels.shape)}")
    if labels.is_floating_point():
        raise ValueError("labels must be an integer tensor")
    dtype = dtype or torch.get_default_dtype()
    d = labels.ndim - 2
    w = _per_sample(t_bg, d, dtype=dtype, device=labels.device).expand(
        labels.shape[0], *labels.shape[1:]
    )
    for label, tk in t.items():
        w = torch.where(labels == label, _per_sample(tk, d, dtype=dtype, device=labels.device), w)
    return gaussian_smooth(w.contiguous(), sigma)


def random_blobs(
    shape: Sequence[int],
    n: int,
    *,
    t: tuple[float, float] = (-0.25, 1.25),
    sigma: float | tuple[float, float] = 8.0,
    t_bg: Time = 0.0,
    within: Tensor | None = None,
    batch: int = 1,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """``n`` Gaussian blobs per sample with times drawn from ``U(t)``, on a ``t_bg`` background.

    Each blob has a centre drawn uniformly from the grid (or from the non-zero voxels of
    ``within``, ``(B or 1, 1, *shape)``), a width ``sigma`` in voxels (fixed or ``U(sigma)``) and a
    time ``t_k``. The blobs are blended as ``w = t_bg + Σ g_k (t_k - t_bg) / max(1, Σ g_k)``, a
    convex combination, so overlapping blobs never push ``w`` outside the range of their times.
    """
    dtype = dtype or torch.get_default_dtype()
    d = len(shape)
    bg = _per_sample(t_bg, d, dtype=dtype, device=device)
    if n == 0:
        return bg.expand(batch, 1, *shape).clone()

    def uniform(lo: float, hi: float, *size: int) -> Tensor:
        u = torch.rand(*size, generator=generator, dtype=torch.float64)
        return (lo + (hi - lo) * u).to(dtype=dtype, device=device)

    if within is None:
        hi = torch.tensor([s - 1 for s in shape], dtype=torch.float64)
        centres = (torch.rand(batch, n, d, generator=generator, dtype=torch.float64) * hi).to(
            dtype=dtype, device=device
        )
    else:
        mask = within.detach().to("cpu").reshape(within.shape[0], -1) != 0
        mask = mask.expand(batch, -1) if mask.shape[0] == 1 else mask
        if mask.shape[0] != batch or not bool(mask.any(1).all()):
            raise ValueError(
                "within must be (1 or batch, 1, *shape) and non-empty for every sample"
            )
        flat = torch.multinomial(mask.double(), n, replacement=True, generator=generator)
        idx = torch.stack(torch.unravel_index(flat, tuple(shape)), dim=-1)
        centres = idx.to(dtype=dtype, device=device)
    widths = (
        uniform(*sigma, batch, n)
        if isinstance(sigma, tuple)
        else torch.full((batch, n), float(sigma), dtype=dtype, device=device)
    )
    times = uniform(*t, batch, n)

    x = identity(shape, dtype=dtype, device=device)  # (1, D, *shape)
    num = torch.zeros(batch, 1, *shape, dtype=dtype, device=device)
    total = torch.zeros_like(num)
    view = (batch, 1, *([1] * d))
    for k in range(n):
        r2 = ((x - centres[:, k].view(batch, d, *([1] * d))) ** 2).sum(1, keepdim=True)
        g = torch.exp(-0.5 * r2 / widths[:, k].view(view) ** 2)
        num = num + g * times[:, k].view(view)
        total = total + g
    return bg + (num - total * bg) / total.clamp_min(1.0)


def rate(w: Tensor, v: Tensor) -> Tensor:
    """Per-sample ``max |∇w · v|`` (voxel units, so dimensionless), shape ``(B,)``.

    The Lagrangian map of :func:`drift.flow.integrate` is invertible only while ``∇w · v < 1``
    everywhere. The absolute value makes the bound hold for both ``v`` and ``-v``.
    """
    d = _check_field(v)
    if w.ndim != v.ndim or w.shape[1] != 1 or w.shape[2:] != v.shape[2:]:
        raise ValueError(f"w must be (B, 1, *spatial) matching v, got shape {tuple(w.shape)}")
    grad = torch.cat(torch.gradient(w, dim=tuple(range(2, 2 + d))), dim=1)
    return (grad * v).sum(1).abs().flatten(1).amax(1)


def limit_rate(
    w: Tensor,
    v: Tensor,
    max_rate: float = MAX_RATE,
    *,
    max_sigma: float | None = None,
) -> Tensor:
    """Make ``rate(w, v) <= max_rate`` for every sample.

    Samples over the limit are smoothed with a growing Gaussian (``sigma = 1, √2, 2, …`` voxels,
    up to ``max_sigma``, default a quarter of the largest grid size), which keeps the times in the
    cores of large regions. If that is not enough, their deviation from the per-sample mean of ``w``
    is scaled down until the bound holds, which shrinks the times towards that mean.
    """
    r = rate(w, v)
    if bool((r <= max_rate).all()):
        return w
    b = max(w.shape[0], v.shape[0])
    w0 = w.expand(b, *w.shape[1:])
    v = v.expand(b, *v.shape[1:])
    out = w0.clone()
    limit = max_sigma if max_sigma is not None else max(w.shape[2:]) / 4
    sigma = 1.0
    bad = rate(out, v) > max_rate
    while bool(bad.any()) and sigma <= limit:
        out[bad] = gaussian_smooth(w0[bad], sigma)
        bad = rate(out, v) > max_rate
        sigma *= 2**0.5
    if bool(bad.any()):
        r = rate(out[bad], v[bad])
        mean = out[bad].flatten(1).mean(1).view(-1, *([1] * (w.ndim - 1)))
        scale = (max_rate * (1 - 1e-6) / r).view_as(mean)  # stay strictly inside the bound
        out[bad] = mean + (out[bad] - mean) * scale
    return out
