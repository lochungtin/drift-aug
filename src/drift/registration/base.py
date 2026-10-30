"""The protocol every registration backend implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from torch import Tensor

from drift.jacobian import JacobianStats
from drift.space import ImageSpace


@dataclass(frozen=True)
class RegistrationResult:
    """Affine pre-alignment plus a non-rigid SVF, both relative to the reference image A.

    - ``affine``: ``(D+1, D+1)`` world-mm matrix mapping reference world points to floating world
      points (NiftyReg's ``Affine * Reference = Floating``).
    - ``velocity``: ``(1, D, *space.shape)`` SVF in voxel units on the reference grid.
      ``exp(velocity)`` is the pull-back map from the reference grid onto the floating image
      after affine alignment: ``B_aligned ∘ exp(v) ≈ A``, and so ``A ∘ exp(-v) ≈ B_aligned``.
    - ``qc``: Jacobian statistics of ``exp(velocity)`` plus backend-specific scalars.
    """

    space: ImageSpace
    affine: npt.NDArray[np.float64]
    velocity: Tensor
    jacobian: JacobianStats
    qc: dict[str, float] = field(default_factory=dict)


class Registration(Protocol):
    def register(
        self,
        reference: str | Path,
        floating: str | Path,
        *,
        reference_mask: str | Path | None = None,
        floating_mask: str | Path | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """Register ``floating`` (B) to ``reference`` (A): affine, then a non-rigid SVF.

        ``affine``: skip the affine step and use this one (``affine @ A_world = B_world``), e.g. to
        register several compartments relative to one shared affine.
        """
        ...
