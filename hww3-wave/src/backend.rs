//! What the driver needs from a GPU backend: buffers of one frequency's
//! state `[np, nth]`, one N-scheme sub-step and its transpose.

/// Float type of the kernels (f32 as WW3, f64 for the checks).
pub trait Scalar: Copy + Send + Sync + std::fmt::Debug + 'static {
    const NAME: &'static str;
    fn of_f64(x: f64) -> Self;
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
    pub np: usize,
    pub ne: usize,
    pub nk: usize,
    pub nth: usize,
}

pub trait Sweeper {
    type Buf;
    /// Backend and float type, e.g. `cubecl f32`.
    fn label(&self) -> String;
    fn float_bytes(&self) -> usize;
    fn dims(&self) -> Dims;
    /// `ITER(ik, ith)`.
    fn iter(&self) -> &[u32];
    fn block(&self) -> u32;
    fn set_block(&mut self, b: u32);
    /// Names of the kernel variants this backend has: `fused` (node gather
    /// that recomputes the triangle quantities), `two_phase` (one pass over
    /// triangle and direction, then the node gather), `two_phase_x4` (the
    /// same, four directions per unit).
    fn variants(&self) -> &'static [&'static str];
    fn variant(&self) -> usize;
    fn set_variant(&mut self, v: usize);
    /// Time spent computing the per-triangle coefficients of all frequencies (s).
    fn coef_seconds(&self) -> f64;
    fn upload(&self, v: &[f64]) -> Self::Buf;
    fn download(&self, b: &Self::Buf) -> Vec<f64>;
    fn empty(&self) -> Self::Buf;
    /// One sub-step `out = L_sub u` of frequency `ik` (with WW3's `max(0, .)` if `clip`).
    fn forward(&self, u: &Self::Buf, out: &mut Self::Buf, ik: usize, sub: u32, clip: bool);
    /// One transposed sub-step `out = L_subᵀ y`.
    fn transpose(&self, y: &Self::Buf, out: &mut Self::Buf, ik: usize, sub: u32);
    fn sync(&self);

    fn len(&self) -> usize {
        let d = self.dims();
        d.np * d.nth
    }

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
