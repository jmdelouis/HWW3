//! Burn integration.
//!
//! The states are Burn tensors `[np, nth]` on `Device::cuda(0)`. The N-scheme
//! kernels run on the CubeCL client and buffers behind them, and a sub-step is
//! a custom operation of Burn's autodiff tape whose backward is the transpose
//! kernel: the gradient of `<L x, ȳ>` with respect to `x` is `Lᵀ ȳ`.

use crate::backend::Sweeper;
use crate::device::DeviceGrid;
use burn::tensor::{DType, Device, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{unary, Backward, Ops, OpsKind};
use burn_autodiff::Autodiff;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;
use cubecl::client::Client;
use cubecl::server::Handle;
use std::sync::Arc;

pub type B = CubeBackend;
pub type Ad = Autodiff<CubeBackend>;

/// The CubeCL client behind a Burn device (plain or autodiff).
pub fn client(device: &Device) -> anyhow::Result<Client> {
    let t = Tensor::<1>::zeros([1], device);
    if let Ok(p) = t.clone().try_into_primitive::<Ad>() {
        return Ok(p.primitive().client.clone());
    }
    let p = t.try_into_primitive::<B>().map_err(|e| anyhow::anyhow!("not a CubeCL tensor: {e:?}"))?;
    Ok(p.client.clone())
}

fn wrap(dg: &DeviceGrid<f32>, like: &CubeTensor, h: Handle) -> CubeTensor {
    CubeTensor::new_contiguous(like.client.clone(), like.device.clone(), [dg.np, dg.nth].into(), h, DType::F32)
}

/// `L_sub x` on a Burn primitive (linear part, no clip).
pub fn forward_primitive(dg: &DeviceGrid<f32>, x: &CubeTensor, ik: usize, sub: u32) -> CubeTensor {
    let x = if x.is_contiguous() { x.clone() } else { into_contiguous(x.clone()) };
    let mut out = dg.empty();
    dg.forward(&x.handle, &mut out, ik, sub, false);
    wrap(dg, &x, out)
}

/// `L_subᵀ y` on a Burn primitive.
pub fn transpose_primitive(dg: &DeviceGrid<f32>, y: &CubeTensor, ik: usize, sub: u32) -> CubeTensor {
    let y = if y.is_contiguous() { y.clone() } else { into_contiguous(y.clone()) };
    let mut out = dg.empty();
    dg.transpose(&y.handle, &mut out, ik, sub);
    wrap(dg, &y, out)
}

/// One sub-step of frequency `ik` as an operation of the tape.
#[derive(Clone)]
pub struct SubStep {
    pub dg: Arc<DeviceGrid<f32>>,
    pub ik: usize,
    pub sub: u32,
}

impl std::fmt::Debug for SubStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SubStep(ik {}, sub {})", self.ik, self.sub)
    }
}

#[derive(Debug)]
struct SubStepBackward;

impl Backward<B, 1> for SubStepBackward {
    type State = SubStep;

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let st = ops.state.clone();
        unary::<B, _>(ops.parents, ops.node, grads, |g| transpose_primitive(&st.dg, &g, st.ik, st.sub));
    }
}

/// `L_sub x` for a tensor on an autodiff CUDA device, recorded on the tape.
pub fn substep(x: Tensor<2>, st: SubStep) -> Tensor<2> {
    let xa = x.try_into_primitive::<Ad>().expect("an autodiff CubeCL tensor");
    let out = forward_primitive(&st.dg, xa.primitive(), st.ik, st.sub);
    let res = match SubStepBackward.prepare::<NoCheckpointing>([xa.node()]).compute_bound().stateful() {
        OpsKind::Tracked(prep) => prep.finish(st, out),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    Tensor::from_primitive::<Ad>(res)
}
