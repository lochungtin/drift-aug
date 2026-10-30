# Changelog

## Initial development

First version: diffeomorphic, regionally controllable augmentation for segmentation, based on
inter-patient stationary velocity fields.

### Core
- N-D (2D/3D) PyTorch core: grids and pull-back warping (`grid`), scaling-and-squaring
  exponentiation with automatic step choice and Eulerian / Lagrangian regional flows (`flow`),
  Jacobians and QC statistics (`jacobian`), cubic B-splines (`bspline`), image geometry (`space`).
- Regional weight maps from labels, random blobs or constants, with a guard that keeps
  Lagrangian maps invertible (`weights`).

### Registration
- `torch` backend (default): affine (multi-start, NMI) plus cubic B-spline SVF with NMI / LNCC /
  SSD, NiftyReg-style regularisation and log-Jacobian penalty, on the GPU. Calibrated against
  NiftyReg on labelled breast MRI (Duke) and thoracic MRI (TotalSegmentator). The implementation
  is the `pyniftireg` package (extracted from this project); drift wraps it.
- `niftyreg` backend: wraps `reg_aladin` + `reg_f3d -vel` with convention-checked conversion of
  control-point grids to dense SVFs; the reference implementation.
- Optional compartment registration with velocity blending for sliding interfaces.

### Banks and augmentation
- Site profiles (YAML): `breast_mri`, `thorax_mri`.
- Preprocessing: reorientation, resampling, N4, normalisation, masks.
- `FieldBank`: pairwise SVFs with per-pair QC, partner selection, resumable builds, coarse storage,
  fields served on any grid of the subject's world frame.
- `DriftAugment` with global / regional / blob / mixed time policies and Jacobian / displacement
  rejection rules. `field_spacing` computes the flow on a coarser grid and upsamples it (regional
  Lagrangian flows ~6x faster at 3 mm, mean error < 0.05 voxels).
- Partner features can come from labels or from a label-free foreground volume
  (`partners.feature_source`), so labelled subjects can be paired with unlabelled scans.
- Integrations: MONAI (`DriftAugmentd`), TorchIO (`RandomDrift`), offline nnU-Net example.
- CLI: `drift profiles | register | bank build | bank info | sweep | augment`.
