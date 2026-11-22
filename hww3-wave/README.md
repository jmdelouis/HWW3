# hww3-wave: the GPU wave model of HWW3

`hww3-wave` rewrites in Rust, for one GPU, the parts of WAVEWATCH III that
the HWW3 explicit configuration runs on the HEALPix unstructured grids.
It is the first stage of a differentiable WW3, meant to return the
gradients of the misfit to altimeters and buoys with respect to the
forcing fields and the model parameters.

- **Milestone M0 (done, go).** The PDLIB explicit N scheme of the
  geographic propagation and its transpose, as GPU kernels: 0.93 ms per
  frequency sweep on grid v4 on one A100. The dot-product tests pass to
  round-off, and a sub-step is an operation of Burn's autodiff tape.
- **Milestone M1 (targets met).** `hww3-prop`, the propagation-only model
  (geographic and spectral propagation, currents, water levels, ice,
  boundary spectra, WW3 restarts): 48 h of grid v5 K=4 in 32 s on one A100,
  against 813 s for WW3 on 32 cores, within 0.94 mm RMS in H_s at the median
  node.

## Documentation

- GPU wave model: https://jmdelouis.github.io/HWW3/wave_model.html
- Numerics: https://jmdelouis.github.io/HWW3/wave_numerics.html
- Results of M0 and M1: https://jmdelouis.github.io/HWW3/wave_results.html
- Design and cost: https://jmdelouis.github.io/HWW3/differentiable.html
- Rust API: https://jmdelouis.github.io/HWW3/rust-wave/hww3_wave/
  (or `cargo doc --no-deps --open`)

## Build

```bash
cargo test --no-default-features       # unit tests, no GPU needed

# CubeCL and Burn (default feature), Ifremer hpc cluster (glibc >= 2.34)
module load cuda/13.0.0
cargo build --release

# CUDA C fallback through cudarc, Datarmor (driver limited to CUDA 11.4)
CUDARC_CUDA_VERSION=11040 cargo build --release --no-default-features --features cuda-c
```

## Programs

- `hww3-wave info|bench <grid dir> ...`: geometry and sub-cycling of a grid,
  sweep timings, dot-product tests, check against the CPU loop of WW3
  (milestone M0).
- `hww3-prop run|restart-hs|check-ktp3|check-geo <grid dir> ...`: the
  propagation-only model and its checks (milestone M1).

The job scripts are in `../scripts` (`wave_m0_hpc.slurm`, `wave_m0.pbs`,
`m1_prop_hpc.slurm`, `m1_ww3_prop_hpc.slurm`, `m1_compare_nc.py`).
