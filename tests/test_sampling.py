import pytest
import torch

from drift import sampling as s
from drift.profile import load_profile
from drift.space import ImageSpace
from helpers import identity_like, shape_for, smooth_field


def _labels(ndim: int) -> tuple[torch.Tensor, ImageSpace]:
    """Big box split into label 1 (axis-0 lower half) and 2; small cube of label 3 in a corner."""
    shape = shape_for(ndim)
    x = identity_like(shape)
    lab = (x.abs().amax(1, keepdim=True) <= shape[0] / 3).long()
    lab[x[:, :1] > 0] *= 2
    corner = (slice(None), slice(None)) + (slice(1, 4),) * ndim
    lab[corner] = 3
    return lab, ImageSpace.from_spacing(shape, 1.0)


def _at(w: torch.Tensor, idx: tuple[int, ...]) -> float:
    return float(w[(0, 0, *idx)])


def test_distributions() -> None:
    g = torch.Generator().manual_seed(0)
    u = s.Uniform(-1.0, 2.0).sample(1000, g)
    assert float(u.min()) >= -1.0 and float(u.max()) <= 2.0 and abs(float(u.mean()) - 0.5) < 0.1
    assert torch.all(s.Constant(0.3).sample(4) == 0.3)


def test_global_policy_is_reproducible(ndim: int) -> None:
    lab, space = _labels(ndim)
    pol = s.GlobalPolicy(s.Uniform(0.0, 1.0))
    a = pol.sample(lab, space, torch.Generator().manual_seed(3))
    b = pol.sample(lab, space, torch.Generator().manual_seed(3))
    assert not a.regional and a.w.shape == (1,) and torch.equal(a.w, b.w)


def test_regional_policy_assigns_region_times(ndim: int) -> None:
    lab, space = _labels(ndim)
    pol = s.RegionalPolicy(
        {
            "box": s.Region((1, 2), t=s.Constant(1.5)),
            "corner": s.Region((3,), follow_background=True),
        },
        t_bg=s.Constant(0.25),
        sigma_mm=0.0,
    )
    d = pol.sample(lab, space)
    assert d.regional and d.w.shape == lab.shape
    assert torch.all(d.w[lab == 1] == 1.5) and torch.all(d.w[lab == 2] == 1.5)
    assert torch.all(d.w[(lab == 0) | (lab == 3)] == 0.25)
    assert d.times == {"background": 0.25, "box": 1.5, "corner": 0.25}


def test_regional_split_axis_and_missing_regions(ndim: int) -> None:
    lab, space = _labels(ndim)
    pol = s.RegionalPolicy(
        {"box": s.Region((1, 2), split_axis=0), "absent": s.Region((9,))},
        t=s.Uniform(-1.0, 2.0),
        sigma_mm=0.0,
    )
    d = pol.sample(lab, space, torch.Generator().manual_seed(0))
    assert set(d.times) == {"background", "box[0]", "box[1]"}
    assert d.times["box[0]"] != d.times["box[1]"]
    for label, part in ((1, "box[0]"), (2, "box[1]")):
        vals = d.w[lab == label]
        assert torch.allclose(vals, torch.full_like(vals, d.times[part]))


def test_regional_smoothing_and_no_labels(ndim: int) -> None:
    lab, space = _labels(ndim)
    pol = s.RegionalPolicy(
        {"box": s.Region((1, 2), t=s.Constant(2.0))}, t_bg=s.Constant(0.0), sigma_mm=2.0
    )
    d = pol.sample(lab, space)
    edge = d.w[(lab == 1) | (lab == 2)].min()
    assert 0.0 < float(edge) < 2.0  # smoothed at the box boundary
    centre = tuple(n // 2 for n in space.shape)
    assert _at(d.w, centre) == pytest.approx(2.0, abs=0.05)
    fallback = pol.sample(None, space, torch.Generator().manual_seed(0))
    assert not fallback.regional


def test_blob_and_mix_policies(ndim: int) -> None:
    lab, space = _labels(ndim)
    blobs = s.BlobPolicy(
        n=2, t=s.Uniform(1.0, 2.0), t_bg=s.Constant(0.0), sigma_mm=(2.0, 3.0), within_labels=(3,)
    )
    d = blobs.sample(lab, space, torch.Generator().manual_seed(0))
    assert d.regional and float(d.w.max()) > 0.9
    mix = s.MixPolicy([s.GlobalPolicy(s.Constant(0.7)), blobs], probs=[1.0, 0.0])
    assert float(mix.sample(lab, space).w[0]) == pytest.approx(0.7)


def test_policy_from_breast_profile() -> None:
    p = load_profile("breast_mri")
    pol = s.policy_from_config(p.sampling)
    assert isinstance(pol, s.RegionalPolicy)
    assert pol.regions["lesion"].follow_background and pol.regions["breasts"].split_axis == 0
    rules = s.rules_from_config(p.qc)
    assert any(isinstance(r, s.MaxDisplacement) and r.mm == 40.0 for r in rules)


def test_rules(ndim: int) -> None:
    space = ImageSpace.from_spacing(shape_for(ndim), 2.0)
    u = smooth_field(ndim, max_norm=1.0).float()
    assert bool(s.JacobianBounds(0.3, 3.0).check(u, space).all())
    assert not bool(s.JacobianBounds(0.99, 1.01).check(u, space).any())
    assert bool(s.MaxDisplacement(2.01).check(u, space).all())
    assert not bool(s.MaxDisplacement(1.99).check(u, space).any())
