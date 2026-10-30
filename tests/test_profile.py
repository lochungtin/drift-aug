import dataclasses
from pathlib import Path

import pytest

from drift.profile import SiteProfile, available_profiles, load_profile


def test_bundled_profiles_load_and_round_trip(tmp_path: Path) -> None:
    assert "breast_mri" in available_profiles()
    for name in available_profiles():
        p = load_profile(name)
        assert p.name == name
        assert SiteProfile.from_dict(p.to_dict()) == p
        p.to_yaml(tmp_path / f"{name}.yaml")
        assert load_profile(tmp_path / f"{name}.yaml") == p


def test_breast_profile_contents() -> None:
    p = load_profile("breast_mri")
    assert p.preprocess.orientation == "RAS" and p.preprocess.n4
    assert p.registration.mask is not None and p.registration.mask.labels
    assert p.sampling.regions["breasts"].split_axis == 0
    assert p.sampling.regions["lesion"].follow_background
    params = p.registration.niftyreg_params(threads=2)
    assert params.cp_spacing == 10.0 and params.threads == 2 and params.similarity == "nmi"


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        SiteProfile.from_dict({"name": "x", "sampling": {"tee": [0, 1]}})
    with pytest.raises(ValueError):
        SiteProfile.from_dict({"name": "x", "sampling": {"t": [0, 1, 2]}})
    with pytest.raises(ValueError):
        SiteProfile.from_dict({"name": "x", "sampling": {"policy": "sometimes"}})
    with pytest.raises((TypeError, ValueError)):
        SiteProfile.from_dict({"name": "x", "partners": {"k": "eight"}})
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("no_such_site")


def test_defaults_and_coercion() -> None:
    p = SiteProfile.from_dict(
        {"name": "x", "preprocess": {"spacing": [1, 2, 3]}, "registration": {"cp_spacing": [5, 5]}}
    )
    assert p.preprocess.spacing == (1.0, 2.0, 3.0)
    assert p.registration.cp_spacing == (5.0, 5.0)
    assert p.registration.mask is None and p.store_spacing is None


def test_torch_params_convert_niftyreg_weights() -> None:
    from drift.registration.torchreg import NIFTYREG_WEIGHT_SCALE, TorchReg

    rc = load_profile("breast_mri").registration
    tp = rc.torch_params()
    assert tp.bending_energy == pytest.approx(
        rc.bending_energy * NIFTYREG_WEIGHT_SCALE["bending_energy"]
    )
    assert tp.jacobian_log == pytest.approx(rc.jacobian_log * NIFTYREG_WEIGHT_SCALE["jacobian_log"])
    assert tp.cp_spacing == rc.cp_spacing and tp.similarity == rc.similarity
    torch_rc = dataclasses.replace(rc, backend="torch")
    assert isinstance(torch_rc.make(), TorchReg)
