"""Synthetic fields shared by the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from drift.filters import gaussian_smooth

DT = torch.float64


def shape_for(ndim: int) -> tuple[int, ...]:
    return (48, 48) if ndim == 2 else (32, 32, 32)


def smooth_field(
    ndim: int, *, max_norm: float = 3.0, sigma: float = 4.0, seed: int = 0, batch: int = 1
) -> Tensor:
    """Random smooth SVF ``(batch, D, *shape)`` with ``max‖v‖ == max_norm`` voxels."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(batch, ndim, *shape_for(ndim), generator=g, dtype=DT)
    v = gaussian_smooth(v, sigma)
    return v * (max_norm / v.norm(dim=1).max())


def interior(x: Tensor, margin: int) -> Tensor:
    """Crop ``margin`` voxels from every spatial edge of ``(B, C, *spatial)``."""
    idx = (slice(None), slice(None)) + (slice(margin, -margin),) * (x.ndim - 2)
    return x[idx]


def rotation_svf(shape: tuple[int, ...], angle: float) -> Tensor:
    """SVF whose exponential rotates axes (0, 1) by ``angle`` radians about the grid centre."""
    from drift.grid import identity

    x = identity(shape, dtype=DT)
    c = torch.tensor([(n - 1) / 2 for n in shape], dtype=DT).view(1, -1, *[1] * len(shape))
    r = x - c
    v = torch.zeros_like(x)
    v[:, 0] = -angle * r[:, 1]
    v[:, 1] = angle * r[:, 0]
    return v


def shapes_pair(ndim: int, n: int, *, square: float, circle: float) -> tuple[Tensor, Tensor]:
    """Centred square/cube (half-width ``square``) and circle/sphere (radius ``circle``) masks."""
    from drift.grid import identity

    x = identity((n,) * ndim, dtype=DT) - (n - 1) / 2
    sq = (x.abs().amax(1, keepdim=True) <= square).to(DT)
    ci = (x.norm(dim=1, keepdim=True) <= circle).to(DT)
    return sq, ci


def oblique_affine(spacing: tuple[float, ...], origin: tuple[float, ...]) -> np.ndarray:
    """3D affine with a ~20° rotation, anisotropic spacing and a flipped axis."""
    a, b = np.deg2rad(20.0), np.deg2rad(-10.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    m = np.eye(4)
    m[:3, :3] = rz @ rx @ np.diag(spacing)
    m[:3, 3] = origin
    return m


class FakeRegistration:
    """Registration backend returning ``scale · smooth_field`` (seeded per pair), identity affine.

    ``exp(v)`` pulls B back onto A, so A's content moved by ``exp(-v)`` should look like B.
    """

    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale
        self.calls: list[tuple[str, str]] = []
        self.spaces: list[object] = []  # reference grids seen
        self.masks: list[tuple[object, object]] = []
        self.results: list[object] = []

    def register(
        self, reference, floating, *, reference_mask=None, floating_mask=None, affine=None
    ):  # type: ignore[no-untyped-def]
        from drift.flow import exp
        from drift.io import load_nifti
        from drift.jacobian import jacobian_stats
        from drift.registration.base import RegistrationResult

        _, space = load_nifti(reference)
        self.calls.append((str(reference), str(floating)))
        self.spaces.append(space)
        self.masks.append((reference_mask, floating_mask))
        seed = sum(map(ord, str(floating))) % 1000
        if affine is not None:  # compartment registrations: a different field per mask
            seed = (seed + sum(map(ord, Path(str(reference_mask)).name))) % 1000
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(1, space.ndim, *space.shape, generator=g, dtype=torch.float32)
        v = gaussian_smooth(v, 3.0)
        v = v * (self.scale / v.norm(dim=1).max())
        result = RegistrationResult(
            space=space,
            affine=np.eye(space.ndim + 1) if affine is None else affine,
            velocity=v,
            jacobian=jacobian_stats(exp(v)),
            qc={"nmi": 1.1},
        )
        self.results.append(result)
        return result


def write_subjects(root, ndim: int, n: int = 4):  # type: ignore[no-untyped-def]
    """``n`` subjects (image + labels NIfTI) with growing centred boxes.

    Returns ``{id: (image path, labels path)}``.
    """
    from drift.io import save_nifti
    from drift.space import ImageSpace

    shape = (40, 40) if ndim == 2 else (24, 24, 24)
    space = ImageSpace.from_spacing(shape, 1.5)
    x = identity_like(shape)
    out = {}
    for i in range(n):
        half = shape[0] / 6 + i
        lab = (x.abs().amax(1, keepdim=True) <= half).to(torch.int16)
        lab = lab + ((x[:, :1] > 0) & (lab > 0)).to(torch.int16)  # 1 = left half, 2 = right half
        img = gaussian_smooth((lab > 0).float(), 1.0) + 0.05
        sid = f"s{i}"
        save_nifti(root / f"{sid}_img.nii.gz", img, space)
        save_nifti(root / f"{sid}_lab.nii.gz", lab, space)
        out[sid] = (root / f"{sid}_img.nii.gz", root / f"{sid}_lab.nii.gz")
    return out


def identity_like(shape: tuple[int, ...]) -> Tensor:
    from drift.grid import identity

    return identity(shape) - torch.tensor([(n - 1) / 2 for n in shape]).view(
        1, -1, *[1] * len(shape)
    )
