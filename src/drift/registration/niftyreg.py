"""NiftyReg backend: ``reg_aladin`` affine, then ``reg_f3d -vel`` for the non-rigid SVF.

Pipeline for a reference A and floating B:

1. ``reg_aladin`` gives the affine ``Affine * A_world = B_world``.
2. ``reg_resample`` puts B onto A's grid with that affine (``B_aligned``).
3. ``reg_f3d -vel`` registers ``B_aligned`` to A *without* ``-aff``. With an initial affine,
   NiftyReg splits it into two half-affines around the velocity field, which would put the SVF
   in a half-way frame. Without it the control-point grid carries no affine extensions and the
   transformation is exactly ``exp(v)`` on A's grid.
4. The control-point grid is converted to a dense SVF in A's voxel units
   (:func:`velocity_from_cpp`).

The NiftyReg binaries are found via the ``bin_dir`` argument, then the ``NIFTYREG_BIN``
environment variable, then ``PATH``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from drift import bspline
from drift.flow import exp
from drift.grid import identity
from drift.io import PathLike, require_nibabel, space_from_nifti_affine
from drift.jacobian import jacobian_stats
from drift.registration.base import RegistrationResult
from drift.space import ImageSpace

log = logging.getLogger(__name__)

#: NiftyReg's NREG_TRANS_TYPE value for a cubic B-spline velocity control-point grid.
SPLINE_VEL_GRID = 5

BINARIES = ("reg_aladin", "reg_f3d", "reg_resample", "reg_transform", "reg_measure")


class NiftyRegError(RuntimeError):
    pass


def find_binary(name: str, bin_dir: PathLike | None = None) -> Path:
    """Locate a NiftyReg executable: ``bin_dir``, then ``$NIFTYREG_BIN``, then ``PATH``."""
    for d in (bin_dir, os.environ.get("NIFTYREG_BIN")):
        if d:
            p = Path(d) / name
            if p.is_file() and os.access(p, os.X_OK):
                return p
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"NiftyReg executable {name!r} not found. Build NiftyReg "
        "(https://github.com/KCL-BMEIS/niftyreg) and put its bin directory on PATH, "
        "set NIFTYREG_BIN, or pass bin_dir."
    )


def available(bin_dir: PathLike | None = None) -> bool:
    try:
        for b in BINARIES:
            find_binary(b, bin_dir)
    except FileNotFoundError:
        return False
    return True


def run(name: str, args: Sequence[str | Path], *, bin_dir: PathLike | None = None) -> str:
    """Run a NiftyReg executable and return its stdout; raise :class:`NiftyRegError` on failure."""
    cmd = [str(find_binary(name, bin_dir)), *map(str, args)]
    log.debug("running %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # NiftyReg reports some errors with exit code 0, so check stderr too.
    if proc.returncode != 0 or "[NiftyReg ERROR]" in proc.stderr + proc.stdout:
        tail = (proc.stderr or proc.stdout)[-2000:]
        raise NiftyRegError(f"{name} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def read_affine(path: PathLike, ndim: int = 3) -> npt.NDArray[np.float64]:
    """Read a NiftyReg affine text file (4x4), reduced to ``(ndim+1, ndim+1)`` for 2D."""
    a = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    if ndim == 3:
        return a
    if not (np.allclose(a[:2, 2], 0) and np.allclose(a[2, :2], 0)):
        raise ValueError("2D affine mixes the third axis into the plane")
    out = np.eye(3)
    out[:2, :2] = a[:2, :2]
    out[:2, 2] = a[:2, 3]
    return out


def write_affine(path: PathLike, affine: npt.NDArray[np.float64]) -> None:
    """Write a NiftyReg affine text file; 2D ``(3, 3)`` affines are embedded in 4x4."""
    a = np.asarray(affine, dtype=np.float64)
    if a.shape == (3, 3):
        full = np.eye(4)
        full[:2, :2] = a[:2, :2]
        full[:2, 3] = a[:2, 2]
        a = full
    np.savetxt(path, a)


def velocity_from_cpp(cpp: PathLike, reference: ImageSpace) -> Tensor:
    """Dense SVF ``(1, D, *reference.shape)`` in reference voxel units from a ``-vel`` CPP file.

    The file stores control-point *positions* in world mm on a world-axis-aligned grid with its
    own sform. Cubic B-splines reproduce linear functions, so subtracting each control point's
    own position gives the velocity's control values exactly. These are then evaluated at the
    reference voxels' world positions (which need not be axis-aligned with the control grid) and
    converted from world-mm vectors to reference voxel units.
    """
    img = require_nibabel().load(str(cpp))
    hdr = img.header
    if int(hdr["intent_p1"]) != SPLINE_VEL_GRID:
        raise ValueError(f"{cpp} is not a NiftyReg velocity grid (intent_p1={hdr['intent_p1']})")
    if len(hdr.extensions) > 0:
        raise ValueError(
            f"{cpp} carries affine extensions (f3d was run with -aff); these are not supported. "
            "Resample the floating image with the affine first and run f3d without -aff."
        )
    data = np.asarray(img.dataobj, dtype=np.float64)
    d = data.shape[-1]
    if d != reference.ndim:
        raise ValueError(f"{d}-D control grid for a {reference.ndim}-D reference")
    cp_shape = tuple(int(n) for n in data.shape[:d])
    positions = torch.from_numpy(data.reshape(*cp_shape, d)).movedim(-1, 0)[None]
    cp_space = space_from_nifti_affine(img.affine, cp_shape)

    cp_world = cp_space.voxel_to_world(identity(cp_shape, dtype=torch.float64).movedim(1, -1))
    cp_vel = positions - cp_world.movedim(-1, 1)
    if float(hdr["intent_p2"]) < 0:  # backward grid: NiftyReg scales by -2^n
        cp_vel = -cp_vel

    ref_world = reference.voxel_to_world(
        identity(reference.shape, dtype=torch.float64).movedim(1, -1)
    )
    coords = cp_space.world_to_voxel(ref_world).movedim(-1, 1)
    v_mm = bspline.evaluate(cp_vel, coords)
    return reference.displacement_to_voxel(v_mm)


@dataclass(frozen=True)
class NiftyRegParams:
    """Registration parameters. Spacings follow NiftyReg: mm, or voxels if negative."""

    # reg_aladin
    rigid_only: bool = False
    affine_levels: int = 3
    # reg_f3d
    cp_spacing: float | tuple[float, ...] = -5.0
    bending_energy: float = 0.001
    linear_elasticity: float = 0.01
    jacobian_log: float = 0.0
    similarity: Literal["nmi", "lncc", "ssd"] = "nmi"
    lncc_sigma: float = -5.0
    levels: int = 3
    max_iter: int = 150
    threads: int | None = None
    resample_interp: Literal[0, 1, 3] = 3
    extra_aladin_args: tuple[str, ...] = ()
    extra_f3d_args: tuple[str, ...] = ()

    def f3d_args(self, ndim: int) -> list[str]:
        sp = self.cp_spacing if isinstance(self.cp_spacing, tuple) else (self.cp_spacing,) * ndim
        args = ["-sx", str(sp[0]), "-sy", str(sp[1])]
        if ndim == 3:
            args += ["-sz", str(sp[2])]
        args += ["-be", str(self.bending_energy), "-le", str(self.linear_elasticity)]
        args += [
            "-jl",
            str(self.jacobian_log),
            "-ln",
            str(self.levels),
            "-maxit",
            str(self.max_iter),
        ]
        if self.similarity == "lncc":
            args += ["--lncc", str(self.lncc_sigma)]
        elif self.similarity == "ssd":
            args += ["--ssd"]
        return args + list(self.extra_f3d_args)

    def aladin_args(self) -> list[str]:
        args = ["-ln", str(self.affine_levels)]
        if self.rigid_only:
            args.append("-rigOnly")
        return args + list(self.extra_aladin_args)

    def omp(self) -> list[str]:
        return [] if self.threads is None else ["-omp", str(self.threads)]


@dataclass
class NiftyReg:
    """NiftyReg registration backend (implements :class:`~drift.registration.Registration`).

    ``workdir``: if set, intermediate files for each call are kept in a fresh subdirectory of it;
    otherwise a temporary directory is used and deleted.
    """

    params: NiftyRegParams = field(default_factory=NiftyRegParams)
    bin_dir: PathLike | None = None
    workdir: PathLike | None = None

    def _run(self, name: str, args: Sequence[str | Path]) -> str:
        return run(name, [*args, *self.params.omp()], bin_dir=self.bin_dir)

    def nmi(self, reference: PathLike, floating: PathLike, out_dir: Path) -> float:
        out = out_dir / f"nmi_{Path(floating).name}.txt"
        run(
            "reg_measure",
            ["-ref", reference, "-flo", floating, "-nmi", "-out", out],
            bin_dir=self.bin_dir,
        )
        return float(out.read_text().split()[0])

    def register(
        self,
        reference: str | Path,
        floating: str | Path,
        *,
        reference_mask: str | Path | None = None,
        floating_mask: str | Path | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        if self.workdir is None:
            with tempfile.TemporaryDirectory(prefix="drift_niftyreg_") as tmp:
                return self._register(
                    Path(tmp), reference, floating, reference_mask, floating_mask, affine
                )
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=f"{Path(floating).name.split('.')[0]}_", dir=self.workdir)
        return self._register(Path(tmp), reference, floating, reference_mask, floating_mask, affine)

    def _register(
        self,
        tmp: Path,
        reference: PathLike,
        floating: PathLike,
        reference_mask: PathLike | None,
        floating_mask: PathLike | None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        ref_img = require_nibabel().load(str(reference))
        ref_shape = tuple(int(n) for n in ref_img.shape[:3])
        if len(ref_shape) == 3 and ref_shape[2] == 1:
            ref_shape = ref_shape[:2]
        space = space_from_nifti_affine(ref_img.affine, ref_shape)
        p = self.params
        rmask = ["-rmask", str(reference_mask)] if reference_mask is not None else []

        aff = tmp / "affine.txt"
        fmask = ["-fmask", str(floating_mask)] if floating_mask is not None else []
        if affine is not None:
            write_affine(aff, affine)
        else:
            aladin = [
                "-ref",
                reference,
                "-flo",
                floating,
                "-aff",
                aff,
                "-res",
                tmp / "aladin_res.nii.gz",
            ]
            self._run("reg_aladin", [*aladin, *rmask, *fmask, *p.aladin_args(), "-voff"])

        aligned = tmp / "floating_aligned.nii.gz"
        resample = ["-ref", reference, "-trans", aff, "-pad", "0"]
        self._run(
            "reg_resample",
            [
                *resample,
                "-flo",
                floating,
                "-res",
                aligned,
                "-inter",
                str(p.resample_interp),
                "-voff",
            ],
        )
        f3d_fmask: list[str | Path] = []
        if floating_mask is not None:
            aligned_mask = tmp / "floating_mask_aligned.nii.gz"
            self._run(
                "reg_resample",
                [*resample, "-flo", floating_mask, "-res", aligned_mask, "-inter", "0", "-voff"],
            )
            f3d_fmask = ["-fmask", aligned_mask]

        cpp = tmp / "cpp.nii.gz"
        res = tmp / "f3d_res.nii.gz"
        f3d = ["-ref", reference, "-flo", aligned, "-vel", "-cpp", cpp, "-res", res]
        self._run("reg_f3d", [*f3d, *rmask, *f3d_fmask, *p.f3d_args(space.ndim), "-voff"])

        v = velocity_from_cpp(cpp, space).to(torch.float32)
        qc = {"nmi_affine": self.nmi(reference, aligned, tmp), "nmi": self.nmi(reference, res, tmp)}
        return RegistrationResult(
            space=space,
            affine=read_affine(aff, space.ndim),
            velocity=v,
            jacobian=jacobian_stats(exp(v)),
            qc=qc,
        )
