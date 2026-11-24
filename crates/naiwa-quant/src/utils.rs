//! CUDA pointer slicing helpers and GLU activation utilities.

use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr},
};

/// Activation function selector for fused GLU / MMQ / MMVQ operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GluActivationType {
    Silu = 0,
    Gelu = 1,
    Relu = 2,
    GeluErf = 3,
    Sigmoid = 4,
}

/// Apply `act(gate) * up` using standard `candle_nn` ops.
pub fn fused_glu(gate: &Tensor, up: &Tensor, activation: GluActivationType) -> Result<Tensor> {
    let act_gate = match activation {
        GluActivationType::Silu => candle_nn::ops::silu(gate)?,
        GluActivationType::Gelu => gate.gelu()?,
        GluActivationType::Relu => gate.relu()?,
        GluActivationType::GeluErf => gate.gelu_erf()?,
        GluActivationType::Sigmoid => candle_nn::ops::sigmoid(gate)?,
    };
    act_gate.mul(up)
}

#[cfg(feature = "cuda")]
pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

#[cfg(feature = "cuda")]
pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(feature = "cuda")]
pub fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}
