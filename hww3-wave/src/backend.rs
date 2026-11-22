//! What the driver needs from a GPU backend: buffers of one frequency's
//! state `[np, nth]`, one N-scheme sub-step and its transpose.

/// Float type of the kernels (f32 as WW3, f64 for the checks).
pub trait Scalar: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// `"f32"` or `"f64"`.
    const NAME: &'static str;
    /// Conversion from `f64` (rounded for `f32`).
    fn of_f64(x: f64) -> Self;
    /// Conversion to `f64`.
    fn as_f64(self) -> f64;
}

impl Scalar for f32 {
    const NAME: &'static str = "f32";
    fn of_f64(x: f64) -> Self {
        x as f32
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl Scalar for f64 {
    const NAME: &'static str = "f64";
    fn of_f64(x: f64) -> Self {
        x
    }
    fn as_f64(self) -> f64 {
        self
    }
}

/// Sizes shared by the backends.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    /// Number of nodes.
    pub np: usize,
    /// Number of triangles.
    pub ne: usize,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
}

/// A GPU backend of the geographic N scheme: buffers of one frequency's
/// state `[np, nth]` (directions contiguous), the forward sub-step and its
/// transpose. The default methods chain the sub-steps of a 600 s step.
pub trait Sweeper {
    /// Device buffer of one frequency's state.
    type Buf;
    /// Backend and float type, e.g. `cubecl f32`.
    fn label(&self) -> String;
    /// Bytes per value (4 or 8).
    fn float_bytes(&self) -> usize;
    /// Sizes of the grid and the spectrum.
    fn dims(&self) -> Dims;
    /// `ITER(ik, ith)`.
    fn iter(&self) -> &[u32];
    /// Units per block of the kernel launches.
    fn block(&self) -> u32;
    /// Set the units per block.
    fn set_block(&mut self, b: u32);
    /// Names of the kernel variants this backend has: `fused` (node gather
    /// that recomputes the triangle quantities), `two_phase` (one pass over
    /// triangle and direction, then the node gather), `two_phase_x4` (the
    /// same, four directions per unit).
    fn variants(&self) -> &'static [&'static str];
    /// Index of the variant in use, in [`Sweeper::variants`].
    fn variant(&self) -> usize;
    /// Select a variant.
    fn set_variant(&mut self, v: usize);
    /// Time spent computing the per-triangle coefficients of all frequencies (s).
    fn coef_seconds(&self) -> f64;
    /// Copy a host state `[np nth]` to a new device buffer.
    fn upload(&self, v: &[f64]) -> Self::Buf;
    /// Copy a device buffer to the host.
    fn download(&self, b: &Self::Buf) -> Vec<f64>;
    /// A new device buffer of `np nth` values.
    fn empty(&self) -> Self::Buf;
    /// One sub-step `out = L_sub u` of frequency `ik` (with WW3's `max(0, .)` if `clip`).
    fn forward(&self, u: &Self::Buf, out: &mut Self::Buf, ik: usize, sub: u32, clip: bool);
    /// One transposed sub-step `out = L_subᵀ y`.
    fn transpose(&self, y: &Self::Buf, out: &mut Self::Buf, ik: usize, sub: u32);
    /// Wait for the launched kernels to finish.
    fn sync(&self);

    /// Values in one frequency's state, `np nth`.
    fn len(&self) -> usize {
        let d = self.dims();
        d.np * d.nth
    }

    /// Sub-steps of one 600 s step of frequency `ik`: `max_θ ITER(ik, θ)`.
    fn max_iter(&self, ik: usize) -> u32 {
        let nth = self.dims().nth;
        *self.iter()[ik * nth..(ik + 1) * nth].iter().max().unwrap()
    }

    /// The `max_k ITER(ik, k)` sub-steps of one 600 s step of frequency `ik`
    /// (each direction stops after its own `ITER`); returns (result, scratch).
    fn step(&self, a: Self::Buf, b: Self::Buf, ik: usize, clip: bool) -> (Self::Buf, Self::Buf) {
        let (mut a, mut b) = (a, b);
        for sub in 0..self.max_iter(ik) {
            self.forward(&a, &mut b, ik, sub, clip);
            std::mem::swap(&mut a, &mut b);
        }
        (a, b)
    }

    /// Transpose of [`Self::step`]: transposed sub-steps in reverse order.
    fn step_transpose(&self, a: Self::Buf, b: Self::Buf, ik: usize) -> (Self::Buf, Self::Buf) {
        let (mut a, mut b) = (a, b);
        for sub in (0..self.max_iter(ik)).rev() {
            self.transpose(&a, &mut b, ik, sub);
            std::mem::swap(&mut a, &mut b);
        }
        (a, b)
    }
}
