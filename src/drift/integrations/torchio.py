"""TorchIO adapter: :class:`RandomDrift`, a transform on ``tio.Subject``.

``ScalarImage``s are warped as images and ``LabelMap``s as label maps, all with the same
deformation. Each image's ``affine`` (voxel -> RAS world mm, as in NIfTI) gives its grid, so the
subject may be at any spacing of its world frame. Apply it before patch sampling.
"""

from __future__ import annotations

from typing import Any

import torchio as tio

from drift.augment import DriftAugment
from drift.bank import FieldBank
from drift.sampling import Policy
from drift.space import ImageSpace

_SPACE_KEY = "__drift_space__"


class RandomDrift(tio.Transform):
    """DRIFT augmentation for TorchIO.

    The subject id in the bank is read from ``subject[subject_key]``. ``p``, ``include`` and
    ``exclude`` are TorchIO's usual transform arguments; other keyword arguments (``lagrangian``,
    ``rules``, ``max_retries``, ``seed``, ``device``, ``return_field``, …) go to
    :class:`~drift.augment.DriftAugment`. The partner, times and acceptance of the last call are
    stored in ``subject["drift"]``.
    """

    def __init__(
        self,
        bank: FieldBank,
        policy: Policy | None = None,
        *,
        subject_key: str = "subject",
        p: float = 1.0,
        include: Any = None,
        exclude: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, include=include, exclude=exclude)
        self.subject_key = subject_key
        self.augment = DriftAugment(
            bank, policy, subject_key=subject_key, space_key=_SPACE_KEY, **kwargs
        )

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        images = subject.get_images_dict(
            intensity_only=False, include=self.include, exclude=self.exclude
        )
        if not images:
            return subject
        image_keys = [n for n, im in images.items() if im.type == tio.INTENSITY]
        label_keys = [n for n, im in images.items() if im.type == tio.LABEL]
        self.augment.image_keys = tuple(image_keys)
        self.augment.label_keys = tuple(label_keys)
        first = next(iter(images.values()))
        space = ImageSpace(tuple(int(n) for n in first.spatial_shape), first.affine)
        sample: dict[str, Any] = {
            self.subject_key: subject[self.subject_key],
            _SPACE_KEY: space,
            **{n: im.data for n, im in images.items()},
        }
        out = self.augment(sample)
        for n, im in images.items():
            im.set_data(out[n])
        subject["drift"] = out["drift"]
        return subject
