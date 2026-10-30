"""Site profiles: every anatomy-specific choice, bundled as a YAML preset.

The core library only deals with integer labels and numbers. A :class:`SiteProfile` names the
labels and decides preprocessing, the registration mask and parameters, how labels are grouped
into regions with their own ``t`` ranges, partner selection and QC bounds. Presets live in
``drift/profiles/*.yaml``; load one by name (``load_profile("breast_mri")``) or by path.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from drift.registration.niftyreg import NiftyRegParams

if TYPE_CHECKING:
    from drift.registration.base import Registration
    from drift.registration.torchreg import TorchRegParams

Range = tuple[float, float]


@dataclass(frozen=True)
class PreprocessConfig:
    """``orientation``: target axis codes (e.g. ``"RAS"``) or None to keep the stored order.
    ``spacing``: target voxel size in mm, or None to keep it. ``normalize``: percentiles mapped to
    ``[0, 1]``, or None. ``n4``: N4 bias-field correction (needs SimpleITK)."""

    orientation: str | None = "RAS"
    spacing: tuple[float, ...] | None = None
    n4: bool = False
    normalize: Range | None = (1.0, 99.0)
    clip: bool = True


@dataclass(frozen=True)
class MaskConfig:
    """Registration mask: the union of ``labels`` if the subject has labels, else a foreground mask
    of the normalised image (``threshold``: ``"otsu"`` or a value, after smoothing by
    ``smooth_mm`` and opening by ``open_mm``; see :func:`drift.preprocess.foreground_mask`);
    then dilated by ``dilate_mm``."""

    labels: tuple[int, ...] = ()
    threshold: float | Literal["otsu"] | None = None
    smooth_mm: float = 2.0
    open_mm: float = 3.0
    dilate_mm: float = 0.0


@dataclass(frozen=True)
class CompartmentConfig:
    """A region registered on its own (for sliding interfaces): the union of ``labels``, or the
    rest of the body when ``labels`` is empty. Its registration masks are dilated by
    ``dilate_mm``."""

    labels: tuple[int, ...] = ()
    dilate_mm: float = 5.0


@dataclass(frozen=True)
class RegistrationConfig:
    """Registration parameters; spacings follow NiftyReg (mm, or voxels if negative).

    ``backend``: ``torch`` (the default: PyTorch reimplementation, GPU if available; ``device``
    picks one explicitly) or ``niftyreg`` (the reference; external binaries, CPU).
    """

    backend: Literal["niftyreg", "torch"] = "torch"
    rigid_only: bool = False
    similarity: Literal["nmi", "lncc", "ssd"] = "nmi"
    cp_spacing: float | tuple[float, ...] = -5.0
    bending_energy: float = 0.001
    linear_elasticity: float = 0.01
    jacobian_log: float = 0.0
    levels: int = 3
    max_iter: int = 150
    mask: MaskConfig | None = None
    #: Register on images resampled to this spacing (mm); None uses the images as they are.
    spacing: float | None = None
    #: Separately registered regions whose velocities are blended (see FieldBank); needs labels.
    compartments: dict[str, CompartmentConfig] = field(default_factory=dict)
    #: Width (mm) of the transition between compartments (narrow blends strain the tissue).
    blend_sigma_mm: float = 10.0
    device: str | None = None

    def make(self, threads: int | None = None) -> Registration:
        """The registration backend these settings describe."""
        if self.backend == "torch":
            from drift.registration.torchreg import TorchReg

            return TorchReg(self.torch_params())
        from drift.registration.niftyreg import NiftyReg

        return NiftyReg(self.niftyreg_params(threads))

    def torch_params(self) -> TorchRegParams:
        """Torch backend parameters; the weights are in NiftyReg's units and converted."""
        from drift.registration.torchreg import TorchRegParams, from_niftyreg_weights

        w = from_niftyreg_weights(self.bending_energy, self.linear_elasticity, self.jacobian_log)
        return TorchRegParams(
            rigid_only=self.rigid_only,
            cp_spacing=self.cp_spacing,
            bending_energy=w["bending_energy"],
            linear_elasticity=w["linear_elasticity"],
            jacobian_log=w["jacobian_log"],
            similarity=self.similarity,
            levels=self.levels,
            max_iter=self.max_iter,
            device=self.device,
        )

    def niftyreg_params(self, threads: int | None = None) -> NiftyRegParams:
        return NiftyRegParams(
            rigid_only=self.rigid_only,
            cp_spacing=self.cp_spacing,
            bending_energy=self.bending_energy,
            linear_elasticity=self.linear_elasticity,
            jacobian_log=self.jacobian_log,
            similarity=self.similarity,
            levels=self.levels,
            max_iter=self.max_iter,
            threads=threads,
        )


@dataclass(frozen=True)
class RegionConfig:
    """A group of labels that shares one ``t``.

    ``t``: the region's own range; None uses the sampling-wide ``t``. ``follow_background``: use
    the background time instead (e.g. structures whose shape should stay close to the original).
    ``split_axis``: split the region at its centroid along this voxel axis, and give each half
    its own ``t`` (e.g. left/right halves after reorientation)."""

    labels: tuple[int, ...]
    t: Range | None = None
    follow_background: bool = False
    split_axis: int | None = None


@dataclass(frozen=True)
class SamplingConfig:
    """``policy``: ``global`` (one ``t``), ``regional`` (per-region ``t`` from labels) or ``mix``
    (regional with probability ``p_regional``). Regions are painted in order, later ones on top."""

    policy: Literal["global", "regional", "mix"] = "regional"
    t: Range = (-0.25, 1.25)
    t_bg: Range = (0.0, 0.5)
    sigma_mm: float = 6.0
    p_regional: float = 0.5
    lagrangian: bool = True
    regions: dict[str, RegionConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class QCConfig:
    """``pair_*``: bank pairs failing these are excluded. Pair Dice is measured on the union of
    ``pair_dice_labels`` (all non-zero labels if empty) between A's labels warped by the field and
    B's labels after affine alignment. ``det_*`` / ``max_disp_mm``: per-sample rejection bounds used
    by the augmentation."""

    pair_det_min: float = 0.05
    pair_det_max: float = 20.0
    pair_max_fold_fraction: float = 0.0
    pair_min_dice: float | None = None
    pair_dice_labels: tuple[int, ...] = ()
    #: Also record Dice per label of ``pair_dice_labels`` (``dice_<label>``), for analysis only.
    pair_dice_each: bool = False
    det_min: float = 0.3
    det_max: float = 3.0
    max_disp_mm: float | None = None


@dataclass(frozen=True)
class PartnerConfig:
    """``k`` partners per subject, chosen at ``random`` or as the most ``similar`` by a volume.

    ``feature_source``: ``labels`` uses the volume of ``feature_labels`` (or of the registration
    mask when a subject has no labels); ``foreground`` uses an Otsu foreground mask of the image
    for every subject, which keeps labelled and unlabelled subjects comparable (e.g. labelled
    training subjects with unlabelled partners)."""

    k: int = 8
    method: Literal["random", "similar"] = "similar"
    feature_labels: tuple[int, ...] = ()
    feature_source: Literal["labels", "foreground"] = "labels"


@dataclass(frozen=True)
class SiteProfile:
    name: str
    description: str = ""
    labels: dict[str, int] = field(default_factory=dict)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    partners: PartnerConfig = field(default_factory=PartnerConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    #: Bank storage resolution in mm (fields are smooth); None stores at the image resolution.
    store_spacing: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SiteProfile:
        return _build(cls, data, "profile")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SiteProfile:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(dataclasses.asdict(self))
        return out

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def available_profiles() -> list[str]:
    """Names of the bundled presets."""
    root = resources.files("drift") / "profiles"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".yaml"))


def load_profile(name_or_path: str | Path) -> SiteProfile:
    """Load a bundled preset by name, or any profile YAML by path."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return SiteProfile.from_yaml(p)
    res = resources.files("drift") / "profiles" / f"{name_or_path}.yaml"
    if not res.is_file():
        raise ValueError(
            f"unknown profile {name_or_path!r}; bundled profiles: {available_profiles()}"
        )
    return SiteProfile.from_dict(yaml.safe_load(res.read_text()))


# --- dict <-> dataclass ------------------------------------------------------------------------

T = TypeVar("T")


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ValueError(f"{where}: may not be null")
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, where)
            except (TypeError, ValueError) as e:
                errors.append(str(e))
        raise ValueError(f"{where}: {value!r} does not match {tp} ({'; '.join(errors)})")
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        return _build(tp, value, where)
    if origin is Literal:
        if value not in get_args(tp):
            raise ValueError(f"{where}: {value!r} is not one of {get_args(tp)}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, where) for v in value)
        if len(args) != len(value):
            raise ValueError(f"{where}: expected {len(args)} values, got {len(value)}")
        return tuple(_convert(a, v, where) for a, v in zip(args, value, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        kt, vt = get_args(tp)
        return {_convert(kt, k, where): _convert(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp in (int, str, bool):
        if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
            raise TypeError(f"{where}: expected {tp.__name__}, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported type {tp}")  # pragma: no cover


def _build(cls: type[T], data: dict[str, Any], where: str) -> T:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)
