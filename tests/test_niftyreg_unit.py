"""NiftyReg adapter checks that don't need the binaries."""

from pathlib import Path

import numpy as np
import pytest

from drift.space import ImageSpace

nib = pytest.importorskip("nibabel")
from drift.registration.niftyreg import find_binary, velocity_from_cpp  # noqa: E402


def test_rejects_non_velocity_grid(tmp_path: Path) -> None:
    img = nib.Nifti1Image(np.zeros((4, 4, 1, 1, 2), np.float32), np.eye(4))
    nib.save(img, tmp_path / "x.nii.gz")
    with pytest.raises(ValueError, match="not a NiftyReg velocity grid"):
        velocity_from_cpp(tmp_path / "x.nii.gz", ImageSpace.from_spacing((4, 4)))


def test_missing_binary_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIFTYREG_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError, match="NIFTYREG_BIN"):
        find_binary("reg_f3d")


@pytest.mark.parametrize("ndim", [2, 3])
def test_write_read_affine_round_trip(tmp_path: Path, ndim: int) -> None:
    from drift.registration.niftyreg import read_affine, write_affine

    a = np.eye(ndim + 1)
    a[:ndim, :ndim] = [[1.1, 0.2, 0.0][:ndim], [-0.1, 0.9, 0.05][:ndim], [0.0, 0.1, 1.0][:ndim]][
        :ndim
    ]
    a[:ndim, ndim] = [3.0, -2.0, 1.5][:ndim]
    write_affine(tmp_path / "aff.txt", a)
    assert np.allclose(read_affine(tmp_path / "aff.txt", ndim), a)
