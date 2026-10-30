"""MONAI adapter: :class:`DriftAugmentd`, a randomizable dictionary transform.

Images loaded with MONAI (``LoadImaged``) are ``MetaTensor``s whose ``affine`` maps voxel indices to
RAS world mm, the same convention as NIfTI and :class:`~drift.space.ImageSpace`. The grid is read
from that affine, so images may be at any spacing of the subject's world frame (e.g. after
``Spacingd``), as long as no cropping or reorientation changed the voxel-to-world mapping without
updating the affine. Put ``DriftAugmentd`` before random crops: the field is defined on the whole
image.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Randomizable
from torch import Tensor

from drift.augment import DriftAugment
from drift.bank import FieldBank
from drift.sampling import Policy
from drift.space import ImageSpace

_SPACE_KEY = "__drift_space__"


class DriftAugmentd(Randomizable, MapTransform):
    """Apply :class:`~drift.augment.DriftAugment` to a MONAI data dictionary.

    ``keys`` are all keys to warp; those also in ``label_keys`` are warped as label maps (soft
    one-hot, argmax), the rest as images. ``subject_key`` names the entry holding the subject id
    in the bank. Other keyword arguments (``lagrangian``, ``rules``, ``max_retries``, ``prob``,
    ``device``, ``valid_mask_key``, ``return_field``, …) go to ``DriftAugment``.
    ``set_random_state`` (called by MONAI's ``Compose``/``set_determinism``) reseeds it.
    """

    def __init__(
        self,
        keys: KeysCollection,
        bank: FieldBank,
        policy: Policy | None = None,
        *,
        label_keys: Sequence[str] = ("label",),
        subject_key: str = "subject",
        allow_missing_keys: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        names = [str(k) for k in self.keys]
        labels = [k for k in names if k in set(label_keys)]
        self.augment = DriftAugment(
            bank,
            policy,
            image_keys=[k for k in names if k not in labels],
            label_keys=labels,
            subject_key=subject_key,
            space_key=_SPACE_KEY,
            seed=seed,
            **kwargs,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> DriftAugmentd:
        super().set_random_state(seed, state)
        self.augment.set_seed(int(self.R.randint(2**31 - 1)))
        return self

    def randomize(self, data: Any = None) -> None:
        """Draws happen inside ``DriftAugment`` (seeded through ``set_random_state``)."""

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d: dict[Hashable, Any] = dict(data)
        present = [str(k) for k in self.key_iterator(d)]
        if not present:
            return d
        originals = {k: d[k] for k in present}
        for k in present:
            if isinstance(d[k], MetaTensor):
                d[k] = d[k].as_tensor()
        ref = originals[present[0]]
        if isinstance(ref, MetaTensor):
            affine = np.asarray(ref.affine.detach().cpu().numpy(), dtype=np.float64)
            d[_SPACE_KEY] = ImageSpace(tuple(int(n) for n in ref.shape[1:]), affine)
        out: dict[Hashable, Any] = dict(self.augment(d))  # type: ignore[arg-type]
        out.pop(_SPACE_KEY, None)
        for k, orig in originals.items():
            if isinstance(orig, MetaTensor) and isinstance(out[k], Tensor):
                out[k] = MetaTensor(out[k], meta=orig.meta)
        return out
