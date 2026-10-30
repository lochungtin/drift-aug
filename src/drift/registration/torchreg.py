"""PyTorch registration backend: affine, then a cubic B-spline stationary velocity field.

The implementation lives in `pyNiftiReg <https://github.com/lochungtin/pyNiftiReg>`_ (PyPI
``pyniftireg``), which was extracted from this module. This wrapper keeps drift's interface:
:class:`TorchReg` returns drift's :class:`~drift.registration.base.RegistrationResult` (with
drift's :class:`~drift.space.ImageSpace` and :class:`~drift.jacobian.JacobianStats`), so banks,
profiles and QC are unchanged. Everything else (parameters, weight calibration, similarity and
regularisation terms) is re-exported from ``pyniftireg.registration``.

For a reference A and a floating B: an affine in world mm (``Affine * A_world = B_world``,
NiftyReg's convention; multi-start at the coarsest level), B resampled onto A's grid, then a
B-spline SVF ``v`` on A's voxel grid fitted with NMI / LNCC / SSD plus NiftyReg-style bending
energy, linear elasticity and an optional log-Jacobian penalty. ``exp(v)`` pulls ``B_aligned`` back
onto A. Similarity values in ``qc`` are pyNiftiReg's Parzen NMI, close to but not identical with
NiftyReg's ``reg_measure``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pyniftireg import registration as _pnr
from pyniftireg.registration import (
    NIFTYREG_WEIGHT_SCALE,
    TorchRegParams,
    _refine,
    bending_energy,
    from_niftyreg_weights,
    linear_elasticity,
    lncc,
    nmi,
    ssd,
)

from drift.jacobian import JacobianStats
from drift.registration.base import RegistrationResult
from drift.space import ImageSpace

__all__ = [
    "NIFTYREG_WEIGHT_SCALE",
    "TorchReg",
    "TorchRegParams",
    "_refine",
    "bending_energy",
    "from_niftyreg_weights",
    "linear_elasticity",
    "lncc",
    "nmi",
    "ssd",
]


def _to_drift(r: Any) -> RegistrationResult:
    """pyNiftiReg's result (same fields, its own classes) as drift's types."""
    if isinstance(r, RegistrationResult):
        return r
    j = r.jacobian
    return RegistrationResult(
        space=ImageSpace(r.space.shape, r.space.affine),
        affine=r.affine,
        velocity=r.velocity,
        jacobian=JacobianStats(j.min, j.max, j.mean, j.p01, j.p99, j.fold_fraction),
        qc=dict(r.qc),
    )


@dataclass
class TorchReg(_pnr.TorchReg):
    """PyTorch registration backend (implements :class:`~drift.registration.Registration`).

    pyNiftiReg's ``TorchReg`` with results in drift's types. :meth:`register` takes NIfTI paths
    (or arrays, see pyNiftiReg); :meth:`register_arrays` works on ``(1, 1, *spatial)`` tensors.
    Not thread-safe (``thread_safe = False``): banks run it sequentially.
    """

    def register(  # type: ignore[override]
        self,
        reference: str | Path,
        floating: str | Path,
        *,
        reference_mask: str | Path | None = None,
        floating_mask: str | Path | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """Register ``floating`` (B) to ``reference`` (A): affine, then a non-rigid SVF.

        ``affine``: skip the affine step and use this one (``affine @ A_world = B_world``).
        """
        r = super().register(
            reference,
            floating,
            reference_mask=reference_mask,
            floating_mask=floating_mask,
            affine=affine,
        )
        return _to_drift(r)

    def register_arrays(self, *args: Any, **kwargs: Any) -> RegistrationResult:  # type: ignore[override]
        # pyNiftiReg's register() loads its inputs and calls this, so both paths convert here.
        return _to_drift(super().register_arrays(*args, **kwargs))
