import pytest
import torch

from drift import flow, weights
from drift.grid import identity, warp
from drift.jacobian import det
from drift.space import ImageSpace
from helpers import DT, shape_for, smooth_field


def _ball_labels(ndim: int) -> torch.Tensor:
    """Label 1 = ball, 2 = its inner half, 0 = background."""
    shape = shape_for(ndim)
    x = identity(shape, dtype=DT) - torch.tensor([(n - 1) / 2 for n in shape], dtype=DT).view(
        1, -1, *[1] * ndim
    )
    r = x.norm(dim=1, keepdim=True)
    radius = shape[0] / 4
    return (r <= radius).long() + (r <= radius / 2).long()


def _translation(ndim: int, a: float) -> torch.Tensor:
    v = torch.zeros(1, ndim, *shape_for(ndim), dtype=DT)
    v[:, 0] = a
    return v


def test_sigma_voxels() -> None:
    space = ImageSpace.from_spacing((10, 10, 10), spacing=(0.5, 1.0, 2.0))
    assert weights.sigma_voxels(4.0, space) == pytest.approx((8.0, 4.0, 2.0))


def test_constant(ndim: int) -> None:
    shape = shape_for(ndim)
    w = weights.constant(0.7, shape, dtype=DT)
    assert w.shape == (1, 1, *shape) and torch.all(w == 0.7)
    wb = weights.constant(torch.tensor([0.5, -1.0]), shape, dtype=DT)
    assert wb.shape == (2, 1, *shape)
    assert torch.all(wb[0] == 0.5) and torch.all(wb[1] == -1.0)


def test_from_labels_unsmoothed(ndim: int) -> None:
    seg = _ball_labels(ndim)
    w = weights.from_labels(seg, {1: 2.0, 2: -1.0}, t_bg=0.25, dtype=DT)
    assert torch.all(w[seg == 0] == 0.25)
    assert torch.all(w[seg == 1] == 2.0)
    assert torch.all(w[seg == 2] == -1.0)
    # unlisted labels get the background time
    w = weights.from_labels(seg, {2: 1.0}, t_bg=0.5, dtype=DT)
    assert torch.all(w[seg == 1] == 0.5)


def test_from_labels_per_sample_times(ndim: int) -> None:
    seg = _ball_labels(ndim).expand(2, *_ball_labels(ndim).shape[1:])
    w = weights.from_labels(
        seg, {1: torch.tensor([1.0, 3.0])}, t_bg=torch.tensor([0.0, -0.5]), dtype=DT
    )
    assert torch.all(w[0][seg[0] == 1] == 1.0) and torch.all(w[1][seg[1] == 1] == 3.0)
    assert torch.all(w[0][seg[0] == 0] == 0.0) and torch.all(w[1][seg[1] == 0] == -0.5)


def test_from_labels_smoothing_stays_in_range(ndim: int) -> None:
    seg = _ball_labels(ndim)
    w = weights.from_labels(seg, {1: 2.0, 2: -1.0}, t_bg=0.5, sigma=2.0, dtype=DT)
    assert float(w.min()) >= -1.0 - 1e-9 and float(w.max()) <= 2.0 + 1e-9
    assert float(w.min()) > -1.0 + 1e-3  # actually smoothed
    # a uniform label map is left constant
    flat = weights.from_labels(torch.ones_like(seg), {1: 0.3}, sigma=3.0, dtype=DT)
    assert torch.allclose(flat, torch.full_like(flat, 0.3))


def test_from_labels_validates_input(ndim: int) -> None:
    seg = _ball_labels(ndim)
    with pytest.raises(ValueError):
        weights.from_labels(seg.to(DT), {1: 1.0})
    with pytest.raises(ValueError):
        weights.from_labels(seg[:, 0], {1: 1.0})


def test_random_blobs_deterministic_and_in_range(ndim: int) -> None:
    shape = shape_for(ndim)

    def make(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return weights.random_blobs(
            shape, 6, t=(-0.5, 2.0), sigma=(3.0, 6.0), t_bg=0.25, batch=2, generator=g, dtype=DT
        )

    a, b = make(0), make(0)
    assert a.shape == (2, 1, *shape)
    assert torch.equal(a, b)
    assert not torch.equal(a, make(1))
    assert not torch.equal(a[0], a[1])
    assert float(a.min()) >= -0.5 - 1e-9 and float(a.max()) <= 2.0 + 1e-9


def test_random_blobs_within_mask(ndim: int) -> None:
    shape = shape_for(ndim)
    mask = torch.zeros(1, 1, *shape, dtype=torch.bool)
    corner = (slice(None), slice(None)) + (slice(0, 6),) * ndim
    mask[corner] = True
    g = torch.Generator().manual_seed(0)
    w = weights.random_blobs(shape, 1, t=(1.0, 1.0), sigma=1.5, within=mask, generator=g, dtype=DT)
    peak = torch.unravel_index(w.flatten().argmax(), w.shape)
    assert bool(mask[peak])
    assert float(w.max()) == pytest.approx(1.0)
    empty = weights.random_blobs(shape, 0, t_bg=0.3, batch=3, dtype=DT)
    assert empty.shape == (3, 1, *shape) and torch.all(empty == 0.3)
    with pytest.raises(ValueError):
        weights.random_blobs(shape, 1, within=torch.zeros_like(mask))


def test_rate_analytic(ndim: int) -> None:
    """``w = alpha·x0`` and ``v = a·e0`` give ``∇w·v = a·alpha`` everywhere."""
    a, alpha = -4.0, 0.05
    v = _translation(ndim, a)
    w = alpha * identity(shape_for(ndim), dtype=DT)[:, :1]
    assert float(weights.rate(w, v)) == pytest.approx(abs(a) * alpha)
    with pytest.raises(ValueError):
        weights.rate(w[..., :-1], v)


def test_sharp_labels_fold_lagrangian_until_limited(ndim: int) -> None:
    v = _translation(ndim, -4.0)
    seg = _ball_labels(ndim)
    sharp = weights.from_labels(seg, {1: 1.0, 2: 1.0}, dtype=DT)
    assert float(weights.rate(sharp, v)) > 1.0
    assert float(det(flow.integrate(v, sharp)).min()) <= 0  # tissue collides
    w = weights.limit_rate(sharp, v)
    assert float(weights.rate(w, v)) <= weights.MAX_RATE
    psi = flow.integrate(v, w)
    assert float(det(psi).min()) > 0
    # Lagrangian identity: each point moves by v times the weight at its source
    resid = (psi - v * warp(w, psi, padding_mode="border")).abs().max()
    assert float(resid) < 1e-3


def test_limit_rate_keeps_core_times(ndim: int) -> None:
    """Smoothing (not shrinking) is enough for a big region and a small field."""
    shape = shape_for(ndim)
    seg = torch.zeros(1, 1, *shape, dtype=torch.long)
    seg[:, :, shape[0] // 2 :] = 1
    v = _translation(ndim, 2.0)
    w = weights.limit_rate(weights.from_labels(seg, {1: 1.5}, dtype=DT), v)
    assert float(weights.rate(w, v)) <= weights.MAX_RATE
    assert float(w[:, :, 0].max()) == pytest.approx(0.0, abs=1e-6)
    assert float(w[:, :, -1].min()) == pytest.approx(1.5, abs=1e-6)


def test_limit_rate_falls_back_to_shrinking(ndim: int) -> None:
    v = smooth_field(ndim, max_norm=6.0)
    seg = _ball_labels(ndim)
    w0 = weights.from_labels(seg, {1: 2.0}, t_bg=-1.0, dtype=DT)
    w = weights.limit_rate(w0, v, 0.3, max_sigma=1.0)
    assert float(weights.rate(w, v)) <= 0.3
    assert float(w.max() - w.min()) < 3.0  # times were pulled together


def test_limit_rate_is_per_sample(ndim: int) -> None:
    v = _translation(ndim, 3.0)
    seg = _ball_labels(ndim)
    w = torch.cat(
        [
            weights.from_labels(seg, {1: 1.0}, dtype=DT),
            weights.from_labels(seg, {1: 0.0}, dtype=DT),  # constant: already fine
        ]
    )
    out = weights.limit_rate(w, v)
    assert torch.all(weights.rate(out, v) <= weights.MAX_RATE)
    assert torch.equal(out[1], w[1])
    assert weights.limit_rate(out, v) is out  # nothing to do
