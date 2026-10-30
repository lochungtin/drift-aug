"""Convention round-trips between DRIFT and NiftyReg.

Every test runs real NiftyReg binaries and is skipped when they are not available.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from drift.filters import gaussian_smooth
from drift.flow import exp
from drift.grid import identity, warp, warp_labels
from drift.space import ImageSpace
from helpers import oblique_affine, shapes_pair

nib = pytest.importorskip("nibabel")
from drift.io import load_nifti, save_nifti  # noqa: E402
from drift.registration.niftyreg import (  # noqa: E402
    NiftyReg,
    NiftyRegParams,
    run,
    velocity_from_cpp,
)

pytestmark = pytest.mark.niftyreg

DT = torch.float64
# SSD: both synthetic images share one contrast, and NMI barely moves binary-ish shapes.
FAST = NiftyRegParams(
    cp_spacing=-4.0, similarity="ssd", bending_energy=0.0005, linear_elasticity=0.0
)
BLUR = 1.5


@dataclass
class Case:
    name: str
    space: ImageSpace
    reference: Path
    floating: Path
    ref_mask: Tensor
    flo_mask: Tensor
    work: Path
    result: object

    @property
    def run_dir(self) -> Path:
        (d,) = [p for p in self.work.iterdir() if p.is_dir()]
        return d


def _make_case(name: str, tmp: Path) -> Case:
    if name == "2d":
        sq, ci = shapes_pair(2, 64, square=13.5, circle=15.5)
        sq, ci = gaussian_smooth(sq, BLUR), gaussian_smooth(ci, BLUR)
        a = np.eye(3)
        a[:2, :2] = np.diag([1.5, 1.5])
        a[:2, 2] = [10.0, -20.0]
        space = ImageSpace((64, 64), a)
    else:
        sq, ci = shapes_pair(3, 40, square=9.5, circle=11.5)
        space = ImageSpace((40, 40, 40), oblique_affine((1.2, -1.0, 1.6), (5.0, 6.0, 7.0)))
    sq, ci = gaussian_smooth(sq, BLUR), gaussian_smooth(ci, BLUR)
    # float32: NiftyReg's native type (reg_transform -flow segfaults on 2D float64 inputs)
    save_nifti(tmp / "ref.nii.gz", sq.float(), space)
    save_nifti(tmp / "flo.nii.gz", ci.float(), space)
    work = tmp / "work"
    result = NiftyReg(FAST, workdir=work).register(tmp / "ref.nii.gz", tmp / "flo.nii.gz")
    return Case(name, space, tmp / "ref.nii.gz", tmp / "flo.nii.gz", sq, ci, work, result)


@pytest.fixture(scope="module", params=["2d", "3d_oblique"])
def case(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Case:
    return _make_case(request.param, tmp_path_factory.mktemp(request.param))


def _load_positions(path: Path, space: ImageSpace) -> Tensor:
    """NiftyReg deformation/flow file (world-mm positions) -> (1, D, *shape)."""
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    return torch.from_numpy(data.reshape(*space.shape, space.ndim)).movedim(-1, 0)[None]


def _world(space: ImageSpace, u: Tensor | None = None) -> Tensor:
    x = identity(space.shape, dtype=DT)
    if u is not None:
        x = x + u
    return space.voxel_to_world(x.movedim(1, -1)).movedim(-1, 1)


def _interior(x: Tensor, m: int = 4) -> Tensor:
    return x[(slice(None), slice(None)) + (slice(m, -m),) * (x.ndim - 2)]


def test_dense_velocity_matches_reg_transform_flow(case: Case) -> None:
    cpp = case.run_dir / "cpp.nii.gz"
    flow_file = case.work / "flow.nii.gz"
    run("reg_transform", ["-ref", case.reference, "-flow", cpp, flow_file])
    nr_v_mm = _load_positions(flow_file, case.space) - _world(case.space)
    ours_mm = case.space.displacement_to_world(velocity_from_cpp(cpp, case.space))
    assert float(nr_v_mm.abs().max()) > 0.5  # the registration actually moved something
    assert (ours_mm - nr_v_mm).abs().max() < 1e-3


def test_exp_matches_reg_transform_def(case: Case) -> None:
    cpp = case.run_dir / "cpp.nii.gz"
    def_file = case.work / "def.nii.gz"
    run("reg_transform", ["-ref", case.reference, "-def", cpp, def_file])
    nr_pos = _load_positions(def_file, case.space)
    ours = _world(case.space, exp(velocity_from_cpp(cpp, case.space)))
    err_vox = case.space.displacement_to_voxel(ours - nr_pos).norm(dim=1)
    # different squaring counts and boundary handling; compare away from the border
    assert float(_interior(err_vox[:, None]).max()) < 0.05


def test_warp_matches_reg_resample(case: Case) -> None:
    cpp = case.run_dir / "cpp.nii.gz"
    aligned = case.run_dir / "floating_aligned.nii.gz"
    out = case.work / "resampled.nii.gz"
    run(
        "reg_resample",
        [
            "-ref",
            case.reference,
            "-flo",
            aligned,
            "-trans",
            cpp,
            "-res",
            out,
            "-inter",
            "1",
            "-pad",
            "0",
        ],
    )
    nr, _ = load_nifti(out, dtype=DT)
    flo, _ = load_nifti(aligned, dtype=DT)
    ours = warp(flo, exp(velocity_from_cpp(cpp, case.space)))
    assert float(_interior(ours - nr).abs().max()) < 0.02


def test_sign_convention_and_registration_quality(case: Case) -> None:
    """exp(v) pulls B back onto A, and A ∘ exp(-v) takes on B's shape."""
    r = case.result
    v = r.velocity.to(DT)
    aligned, _ = load_nifti(case.run_dir / "floating_aligned.nii.gz", dtype=DT)
    a_lab, b_lab = (case.ref_mask > 0.5).long(), (aligned > 0.5).long()

    def dice(x: Tensor, y: Tensor) -> float:
        return float(2 * (x * y).sum() / (x.sum() + y.sum()))

    before = dice(a_lab, b_lab)
    assert dice(warp_labels(b_lab, exp(v)), a_lab) > max(0.95, before)
    assert dice(warp_labels(a_lab, exp(v, -1.0)), b_lab) > max(0.95, before)
    # the wrong sign makes things worse, not better
    assert dice(warp_labels(a_lab, exp(v)), b_lab) < before
    assert float(r.jacobian.min[0]) > 0
    assert r.qc["nmi"] > r.qc["nmi_affine"]


def test_affine_sign_is_reference_to_floating(tmp_path: Path) -> None:
    """Affine * reference_world = floating_world."""
    g = torch.Generator().manual_seed(0)
    big = gaussian_smooth(torch.rand(1, 1, 60, 48, 48, generator=g, dtype=DT), 2.0)
    space = ImageSpace.from_spacing((40, 40, 40), 1.0)
    ref = big[:, :, 10:50, 4:44, 4:44]
    flo = big[:, :, 7:47, 4:44, 4:44]  # flo[i] = ref[i - 3]: content moves +3 mm along x
    save_nifti(tmp_path / "ref.nii.gz", ref, space)
    save_nifti(tmp_path / "flo.nii.gz", flo, space)
    r = NiftyReg(NiftyRegParams(rigid_only=True, max_iter=10, levels=1)).register(
        tmp_path / "ref.nii.gz", tmp_path / "flo.nii.gz"
    )
    assert np.allclose(r.affine[:3, 3], [3.0, 0.0, 0.0], atol=0.2)
    assert np.allclose(r.affine[:3, :3], np.eye(3), atol=0.02)


def test_square_to_circle_sweep(case: Case) -> None:
    """Golden demo: r(45°)/r(0°) falls monotonically with t (square √2 -> circle 1 -> diamond)."""
    if case.name != "2d":
        pytest.skip("2D demo")
    v = case.result.velocity.to(DT)
    a = (case.ref_mask > 0.5).to(DT)
    ratios = []
    for t in (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
        img = warp(a, exp(v, -t))[0, 0]
        ratios.append(_diag_over_axis(img))
    print("r(45°)/r(0°) for t in (-1, 0, 0.5, 1, 2, 3):", [round(r, 3) for r in ratios])
    assert ratios[1] > 1.25  # (blur-rounded) square
    assert ratios[3] == pytest.approx(1.0, abs=0.08)  # circle
    assert all(x > y for x, y in itertools.pairwise(ratios)), ratios


def _diag_over_axis(img: Tensor) -> float:
    """Ratio of the shape's extent along the diagonals to its extent along the axes."""
    c = (img.shape[0] - 1) / 2
    s = torch.linspace(0, c, 400, dtype=DT)

    def radius(dx: float, dy: float) -> float:
        d = np.hypot(dx, dy)
        pts = torch.stack([c + s * dx / d, c + s * dy / d], dim=-1)
        vals = img[pts[:, 0].round().long(), pts[:, 1].round().long()]
        inside = (vals > 0.5).nonzero()
        return float(s[inside.max()]) if inside.numel() else 0.0

    diag = np.mean([radius(1, 1), radius(1, -1), radius(-1, 1), radius(-1, -1)])
    axis = np.mean([radius(1, 0), radius(-1, 0), radius(0, 1), radius(0, -1)])
    return float(diag / axis)
