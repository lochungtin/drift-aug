import os
from pathlib import Path

import pytest
import torch

# Use a project-local NiftyReg build (third_party/, gitignored) when NIFTYREG_BIN isn't set.
_LOCAL_NIFTYREG = Path(__file__).parents[1] / "third_party" / "niftyreg" / "install" / "bin"
if "NIFTYREG_BIN" not in os.environ and _LOCAL_NIFTYREG.is_dir():
    os.environ["NIFTYREG_BIN"] = str(_LOCAL_NIFTYREG)

from drift.registration.niftyreg import available  # noqa: E402

HAS_NIFTYREG = available()
HAS_CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_nr = pytest.mark.skip(reason="NiftyReg binaries not found (PATH / NIFTYREG_BIN)")
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    skip_slow = pytest.mark.skip(reason="slow; set DRIFT_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords and os.environ.get("DRIFT_SLOW") != "1":
            item.add_marker(skip_slow)
        if "niftyreg" in item.keywords and not HAS_NIFTYREG:
            item.add_marker(skip_nr)
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def ndim(request: pytest.FixtureRequest) -> int:
    """Spatial dimensionality; math tests take this fixture so they run in both 2D and 3D."""
    return int(request.param)
