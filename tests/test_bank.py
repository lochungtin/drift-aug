from pathlib import Path

import numpy as np
import pytest
import torch

from drift.bank import FieldBank, dice, select_partners
from drift.preprocess import with_spacing
from drift.profile import MaskConfig, PartnerConfig, QCConfig, RegistrationConfig, SiteProfile
from helpers import FakeRegistration, interior, write_subjects


def _profile(**kw: object) -> SiteProfile:
    base: dict[str, object] = {
        "name": "test",
        "registration": RegistrationConfig(mask=MaskConfig(labels=(1, 2), dilate_mm=3.0)),
        "partners": PartnerConfig(k=2, method="similar", feature_labels=(1, 2)),
    }
    base.update(kw)
    return SiteProfile(**base)  # type: ignore[arg-type]


def _bank(tmp_path: Path, ndim: int, **kw: object) -> tuple[FieldBank, FakeRegistration]:
    files = write_subjects(tmp_path, ndim)
    bank = FieldBank(tmp_path / "bank", profile=_profile(**kw))
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    reg = FakeRegistration()
    bank.build(registration=reg)
    return bank, reg


def test_select_partners() -> None:
    feats = {"a": 1.0, "b": 1.1, "c": 4.0, "d": 9.0, "e": None}
    sim = select_partners(feats, 2, "similar")
    assert sim["a"] == ["b", "c"] and sim["d"] == ["c", "b"]
    assert len(sim["e"]) == 2 and "e" not in sim["e"]  # no feature: random
    rnd = select_partners(feats, 3, "random", seed=1)
    assert rnd == select_partners(feats, 3, "random", seed=1)
    assert all(a not in bs and len(bs) == 3 for a, bs in rnd.items())


def test_dice() -> None:
    a = torch.tensor([1, 1, 0, 0])
    assert dice(a, torch.tensor([1, 0, 1, 0])) == 0.5
    assert dice(torch.zeros(3), torch.zeros(3)) == 1.0


def test_build_and_reopen(tmp_path: Path, ndim: int) -> None:
    bank, reg = _bank(tmp_path, ndim)
    assert bank.summary() == {"subjects": 4, "pairs": 8, "passed": 8, "failed": 0}
    assert len(reg.calls) == 8
    # partners are the closest box sizes
    assert bank.partners("s0") == ["s1", "s2"]
    assert bank.mask_path("s0").exists()
    p = bank.pairs[("s0", "s1")]
    assert {"det_min", "fold_fraction", "max_disp_mm", "nmi", "dice", "dice_affine"} <= set(p.qc)
    # resuming skips existing pairs
    assert bank.build(registration=reg) == [] and len(reg.calls) == 8

    again = FieldBank(tmp_path / "bank")
    assert again.profile == bank.profile
    assert again.subjects["s2"].space.shape == bank.subjects["s2"].space.shape
    assert again.pairs == bank.pairs
    with pytest.raises(ValueError):
        FieldBank(tmp_path / "bank", profile=SiteProfile(name="other"))


def test_velocity_sign_and_storage(tmp_path: Path, ndim: int) -> None:
    bank, reg = _bank(tmp_path, ndim)
    a, b = "s0", "s1"
    expected = reg.register(bank.root / bank.subjects[a].image, bank.root / bank.subjects[b].image)
    vel = bank.velocity(a, b)
    assert torch.allclose(vel, -expected.velocity, atol=2e-3)  # float16 storage
    assert bank.velocity(a, b) is vel  # cached


def test_coarse_storage_and_other_grids(tmp_path: Path, ndim: int) -> None:
    bank, reg = _bank(tmp_path, ndim, store_spacing=3.0)
    a, b = "s1", "s2"
    data = np.load(bank.root / bank.pairs[(a, b)].field)
    space = bank.subjects[a].space
    assert tuple(data["shape"]) == with_spacing(space, 3.0).shape
    ref = -reg.register(
        bank.root / bank.subjects[a].image, bank.root / bank.subjects[b].image
    ).velocity
    assert interior(bank.velocity(a, b) - ref, 3).abs().max() < 0.1
    # on a finer grid the same physical displacement is twice as many voxels
    fine = with_spacing(space, 0.75)
    v_fine = bank.velocity(a, b, fine)
    assert v_fine.shape[2:] == fine.shape
    every_other = (slice(None), slice(None)) + (slice(None, None, 2),) * ndim
    assert interior(v_fine[every_other] - 2 * ref, 3).abs().max() < 0.2


def test_qc_failures_are_excluded(tmp_path: Path, ndim: int) -> None:
    bank, _ = _bank(tmp_path, ndim, qc=QCConfig(pair_min_dice=1.01))
    assert bank.summary()["passed"] == 0
    assert bank.partners("s0") == [] and bank.partners("s0", passed_only=False)
    assert all("dice" in r for p in bank.pairs.values() for r in p.reasons)


def test_registration_moves_labels_towards_partner(tmp_path: Path) -> None:
    """With the fake backend the field is random, so only check Dice is recorded per pair."""
    bank, _ = _bank(tmp_path, 2)
    for p in bank.pairs.values():
        assert 0.0 <= p.qc["dice"] <= 1.0 and 0.0 < p.qc["dice_affine"] <= 1.0


def test_rejects_bad_ids_and_misaligned_labels(tmp_path: Path) -> None:
    files = write_subjects(tmp_path, 2, n=1)
    img, lab = files["s0"]
    bank = FieldBank(tmp_path / "bank")
    with pytest.raises(ValueError):
        bank.add_subject("../evil", img, lab)
    from drift.io import load_nifti, save_nifti
    from drift.space import ImageSpace

    x, _ = load_nifti(lab)
    save_nifti(tmp_path / "shifted.nii.gz", x, ImageSpace.from_spacing(x.shape[2:], 2.0))
    with pytest.raises(ValueError):
        bank.add_subject("s0", img, tmp_path / "shifted.nii.gz")


def test_registration_at_coarser_spacing(tmp_path: Path, ndim: int) -> None:
    """Registering on resampled images gives a velocity on A's full grid in A's voxel units."""
    bank, reg = _bank(
        tmp_path,
        ndim,
        registration=RegistrationConfig(spacing=3.0, mask=MaskConfig(labels=(1, 2))),
    )
    space = bank.subjects["s0"].space
    coarse_shape = with_spacing(space, 3.0).shape
    # the backend saw half-resolution images
    assert reg.spaces[0].shape == coarse_shape  # type: ignore[attr-defined]
    vel = bank.velocity("s0", "s1")
    assert vel.shape[2:] == space.shape
    assert bank.pairs[("s0", "s1")].qc["registration_spacing_mm"] == 3.0
    # the fake backend's |v| = 2 coarse voxels = 6 mm = 4 full-resolution voxels
    assert float(vel.norm(dim=1).max()) == pytest.approx(4.0, rel=0.1)


def test_velocity_cache_is_bounded_by_bytes(tmp_path: Path) -> None:
    bank, _ = _bank(tmp_path, 2)
    one = bank.velocity("s0", "s1")
    bank.cache_bytes = int(1.5 * one.numel() * one.element_size())
    bank.velocity("s0", "s2")
    bank.velocity("s1", "s0")
    assert len(bank._cache) == 1  # only the most recent field fits


def test_non_thread_safe_backend_runs_sequentially(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    files = write_subjects(tmp_path, 2, n=3)
    bank = FieldBank(tmp_path / "bank", profile=_profile())
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    reg = FakeRegistration()
    reg.thread_safe = False  # type: ignore[attr-defined]
    with caplog.at_level("WARNING"):
        pairs = bank.build(registration=reg, jobs=4)
    assert len(pairs) == 6 and "sequentially" in caplog.text


class _SameField(FakeRegistration):
    """Every registration of a pair returns the same field, whatever the masks."""

    def register(
        self, reference, floating, *, reference_mask=None, floating_mask=None, affine=None
    ):  # type: ignore[no-untyped-def]
        return super().register(reference, floating)


def _compartment_profile() -> SiteProfile:
    from drift.profile import CompartmentConfig

    return _profile(
        registration=RegistrationConfig(
            mask=MaskConfig(labels=(1, 2), dilate_mm=3.0),
            compartments={"right": CompartmentConfig(labels=(2,)), "rest": CompartmentConfig()},
            blend_sigma_mm=1.5,
        )
    )


def test_compartments_are_a_partition_of_unity(tmp_path: Path, ndim: int) -> None:
    files = write_subjects(tmp_path, ndim, n=2)
    bank = FieldBank(tmp_path / "bank", profile=_compartment_profile())
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    reg = _SameField()
    bank.build(registration=reg)
    assert len(reg.calls) == 2 * 3  # whole body + 2 compartments per pair
    pair = bank.pairs[("s0", "s1")]
    assert pair.qc["compartments"] == 2.0
    base = reg.register(
        bank.root / bank.subjects["s0"].image, bank.root / bank.subjects["s1"].image
    )
    assert torch.allclose(bank.velocity("s0", "s1"), -base.velocity, atol=3e-3)


def test_compartments_follow_their_own_field(tmp_path: Path) -> None:
    files = write_subjects(tmp_path, 2, n=2)
    bank = FieldBank(tmp_path / "bank", profile=_compartment_profile())
    for sid, (img, lab) in files.items():
        bank.add_subject(sid, img, lab)
    reg = FakeRegistration()
    bank.register_pair("s0", "s1", registration=reg)
    base, right, _ = (r.velocity for r in reg.results)  # type: ignore[attr-defined]
    assert reg.masks[1][0] is not None and reg.results[1].affine is not None  # type: ignore[attr-defined]
    v = -bank.velocity("s0", "s1")
    from drift.io import load_nifti

    lab, _ = load_nifti(bank.root / str(bank.subjects["s0"].labels), dtype=torch.int64)
    core = torch.zeros_like(lab, dtype=torch.bool)
    core[..., 22:24, 16:24] = True  # deep inside label 2 (axis 0 > centre)
    assert bool((lab[core] == 2).all())
    assert (v - right).norm(dim=1)[core[:, 0]].max() < 0.15 * float(right.norm(dim=1).max())
    corner = (slice(None), slice(None), slice(0, 3), slice(0, 3))  # air: no compartment
    assert torch.allclose(v[corner], base[corner], atol=3e-3)


def test_per_label_dice(tmp_path: Path) -> None:
    bank, _ = _bank(tmp_path, 2, qc=QCConfig(pair_dice_labels=(1, 2), pair_dice_each=True))
    qc = bank.pairs[("s0", "s1")].qc
    assert {"dice_1", "dice_2", "dice_affine_1", "dice_affine_2"} <= set(qc)
    assert all(0.0 <= qc[k] <= 1.0 for k in ("dice_1", "dice_2"))


def test_foreground_feature_is_label_independent(tmp_path: Path) -> None:
    from drift.profile import PartnerConfig

    files = write_subjects(tmp_path, 2, n=2)
    prof = _profile(partners=PartnerConfig(k=1, feature_source="foreground"))
    with_labels = FieldBank(tmp_path / "a", profile=prof)
    without = FieldBank(tmp_path / "b", profile=prof)
    img, lab = files["s1"]
    f1 = with_labels.add_subject("s1", img, lab).feature
    f2 = without.add_subject("s1", img).feature
    assert f1 is not None and f1 == pytest.approx(f2)
