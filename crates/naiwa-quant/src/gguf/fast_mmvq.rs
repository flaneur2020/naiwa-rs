//! CUDA fast path for GGUF matmul with BF16/F32 activations.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::{
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    GluActivationType,
};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
const MATRIX_ROW_PADDING: usize = 512;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

/// Quant types supported by `mmvq_gguf.cu`.
pub fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::IQ4XS
            | GgmlDType::IQ4NL
            | GgmlDType::Q2_0
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ3_S
            | GgmlDType::IQ2_S
    )
}

/// Maximum flattened batch handled by the CUDA launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    /// M38: which activation is currently packed in `slice`. The slot is already keyed by capacity, so an
    /// activation survives the GEMVs of *other* sizes that run between its two consumers.
    packed: std::cell::Cell<Option<PackedKey>>,
}

/// Identity of the activation whose q8_1 form a workspace holds. The `Arc<Storage>` is held for as long as
/// the entry lives, so its address cannot be recycled by a different live tensor: a stale hit is impossible,
/// only a stale *miss*.
struct PackedKey {
    /// Kept alive so the id can never be reused, and so a hit always means the very same tensor.
    src: Tensor,
    offset: usize,
    k: usize,
    rows: usize,
    dtype: DType,
}

impl PackedKey {
    fn of(xs: &Tensor, k: usize, rows: usize) -> Self {
        let offset = xs.layout().start_offset();
        Self {
            src: xs.clone(),
            offset,
            k,
            rows,
            dtype: xs.dtype(),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self.src.id() == other.src.id()
            && self.offset == other.offset
            && self.k == other.k
            && self.rows == other.rows
            && self.dtype == other.dtype
    }
}

/// M38: `NAIWA_NO_Q8CACHE` is the A/B arm (quantise on every call, the old behaviour).
fn q8_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NAIWA_NO_Q8CACHE").is_none())
}

/// M38 counters: a hit is a launch the cache removed.
fn q8_ticks(hit: bool) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static HIT: AtomicUsize = AtomicUsize::new(0);
    static MISS: AtomicUsize = AtomicUsize::new(0);
    static PRINT: AtomicUsize = AtomicUsize::new(0);
    if std::env::var_os("NAIWA_Q8CACHE_DBG").is_none() {
        return;
    }
    if hit {
        HIT.fetch_add(1, Ordering::Relaxed);
    } else {
        MISS.fetch_add(1, Ordering::Relaxed);
    }
    if PRINT.fetch_add(1, Ordering::Relaxed) % 512 != 0 {
        return;
    }
    let h = HIT.load(Ordering::Relaxed);
    let m = MISS.load(Ordering::Relaxed);
    eprintln!(
        "Q8CACHE hits {h} misses {m} ({:.1}% of calls skipped the quantise launch)",
        100.0 * h as f64 / (h + m).max(1) as f64
    );
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }

    /// Does this workspace already hold the packed form of `key`? True means the quantise launch can be
    /// skipped: the buffer belongs to this (device, stream, thread, capacity) slot and nothing of the same
    /// size has overwritten it since.
    fn holds(&self, key: &PackedKey) -> bool {
        // `Cell::take` gives the claim back by value; put it straight back when it does not match.
        let held = self.slot.packed.take();
        let hit = held.as_ref().is_some_and(|have| have.matches(key));
        self.slot.packed.set(held);
        hit
    }

    /// Record what the following quantise launch puts into this workspace.
    fn set_packed(&self, key: PackedKey) {
        self.slot.packed.set(Some(key));
    }

    /// Any other writer of this workspace must drop the claim, or a later lookup could read stale bytes.
    fn invalidate(&self) {
        self.slot.packed.set(None);
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::cuda::DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

type WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>;

static WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn workspace_ensure<'a>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let workspace_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(capacity)? };
                // CUDA graph replay requires process-stable workspace addresses.
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot {
                    slice,
                    packed: std::cell::Cell::new(None),
                })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    Ok(WorkspaceGuard {
        slot: workspace_mtx.lock().unwrap(),
        stream,
    })
}

// Launcher dispatch by weight and output dtype.

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type FusedQkvLauncher = unsafe extern "C" fn(
    vx_q: *const std::ffi::c_void,
    vx_k: *const std::ffi::c_void,
    vx_v: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    q_dst: *mut std::ffi::c_void,
    k_dst: *mut std::ffi::c_void,
    v_dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_q: i32,
    nrows_k: i32,
    nrows_v: i32,
    stride_col_y: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        GgmlDType::IQ4XS => ffi::launch_mmvq_gguf_iq4xs_bf16_plain,
        GgmlDType::IQ4NL => ffi::launch_mmvq_gguf_iq4nl_bf16_plain,
        GgmlDType::Q2_0 => ffi::launch_mmvq_gguf_q2_0_bf16_plain,
        GgmlDType::IQ3_XXS => ffi::launch_mmvq_gguf_iq3xxs_bf16_plain,
        GgmlDType::IQ2_S => ffi::launch_mmvq_gguf_iq2s_bf16_plain,
        GgmlDType::IQ3_S => ffi::launch_mmvq_gguf_iq3s_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        GgmlDType::IQ4XS => ffi::launch_mmvq_gguf_iq4xs_f16_plain,
        GgmlDType::IQ4NL => ffi::launch_mmvq_gguf_iq4nl_f16_plain,
        GgmlDType::Q2_0 => ffi::launch_mmvq_gguf_q2_0_f16_plain,
        GgmlDType::IQ3_XXS => ffi::launch_mmvq_gguf_iq3xxs_f16_plain,
        GgmlDType::IQ2_S => ffi::launch_mmvq_gguf_iq2s_f16_plain,
        GgmlDType::IQ3_S => ffi::launch_mmvq_gguf_iq3s_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        GgmlDType::IQ4XS => ffi::launch_mmvq_gguf_iq4xs_f32_plain,
        GgmlDType::IQ4NL => ffi::launch_mmvq_gguf_iq4nl_f32_plain,
        GgmlDType::Q2_0 => ffi::launch_mmvq_gguf_q2_0_f32_plain,
        GgmlDType::IQ3_XXS => ffi::launch_mmvq_gguf_iq3xxs_f32_plain,
        GgmlDType::IQ2_S => ffi::launch_mmvq_gguf_iq2s_f32_plain,
        GgmlDType::IQ3_S => ffi::launch_mmvq_gguf_iq3s_f32_plain,
        _ => return None,
    };
    Some(f)
}

fn fused_glu_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedGluLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_bf16_fused_glu),
        (DType::BF16, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_bf16_fused_glu),
        (DType::BF16, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_bf16_fused_glu),
        (DType::BF16, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_bf16_fused_glu),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_glu),
        (DType::F16, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_f16_fused_glu),
        (DType::F16, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_f16_fused_glu),
        (DType::F16, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_f16_fused_glu),
        (DType::F16, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_f16_fused_glu),
        (DType::F16, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_f16_fused_glu),
        (DType::F16, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_f16_fused_glu),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_glu),
        (DType::F32, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_f32_fused_glu),
        (DType::F32, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_f32_fused_glu),
        (DType::F32, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_f32_fused_glu),
        (DType::F32, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_f32_fused_glu),
        (DType::F32, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_f32_fused_glu),
        (DType::F32, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_f32_fused_glu),
        _ => None,
    }
}

pub fn supports_fused_glu(input_ty: DType, dtype: GgmlDType) -> bool {
    fused_glu_launcher(input_ty, dtype).is_some()
}

fn fused_qkv_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedQkvLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_bf16_fused_qkv),
        (DType::BF16, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_bf16_fused_qkv),
        (DType::BF16, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_bf16_fused_qkv),
        (DType::BF16, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_bf16_fused_qkv),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_qkv),
        (DType::F16, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_f16_fused_qkv),
        (DType::F16, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_f16_fused_qkv),
        (DType::F16, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_f16_fused_qkv),
        (DType::F16, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_f16_fused_qkv),
        (DType::F16, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_f16_fused_qkv),
        (DType::F16, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_f16_fused_qkv),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_qkv),
        (DType::F32, GgmlDType::IQ4XS) => Some(ffi::launch_mmvq_gguf_iq4xs_f32_fused_qkv),
        (DType::F32, GgmlDType::IQ4NL) => Some(ffi::launch_mmvq_gguf_iq4nl_f32_fused_qkv),
        (DType::F32, GgmlDType::Q2_0) => Some(ffi::launch_mmvq_gguf_q2_0_f32_fused_qkv),
        (DType::F32, GgmlDType::IQ3_XXS) => Some(ffi::launch_mmvq_gguf_iq3xxs_f32_fused_qkv),
        (DType::F32, GgmlDType::IQ2_S) => Some(ffi::launch_mmvq_gguf_iq2s_f32_fused_qkv),
        (DType::F32, GgmlDType::IQ3_S) => Some(ffi::launch_mmvq_gguf_iq3s_f32_fused_qkv),
        _ => None,
    }
}

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same CUDA device.
///
/// The product of the leading input dimensions must be in `1..=8`.
///
/// Output has the same leading dimensions as `xs` with the last axis replaced
/// by `w.shape().dims2()?.0` (nrows of the weight).
///
/// The output dtype matches the input dtype (BF16 → BF16, F16 → F16, F32 → F32).
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("fast_mmvq: weight must live on CUDA");
    };
    if !xs.device().same_device(&w.device()) {
        candle_core::bail!("fast_mmvq: input and weight are on different devices");
    }
    let (nrows, ncols) = w.shape().dims2()?;

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("fast_mmvq: input dtype must be BF16, F16, or F32, got {input_ty:?}");
    }

    // M38: which projection is this? Counted per (dtype, k, nrows, rows) so the launch-count work starts
    // from the model's actual call sites instead of a guess. `NAIWA_MMVQ_DBG=1` only; the check is one
    // relaxed atomic load on the hot path.
    mmvq_dbg::record(dtype, k, nrows, b_size);

    // M38: the packed form depends only on the values, so the identity of the source tensor is enough to
    // know whether this workspace already holds it. Taken before `contiguous`, so a non-contiguous input
    // that is quantised twice is served from the cache rather than copied and re-quantised.
    let cache_key = q8_cache().then(|| PackedKey::of(xs, k, b_size));

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let packed_ready = cache_key.as_ref().is_some_and(|key| workspace.holds(key));
    q8_ticks(packed_ready);
    if !packed_ready {
        if let Some(key) = cache_key {
            workspace.set_packed(key);
        }
    }
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    if !packed_ready {
                        ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                            xs_ptr as *const std::ffi::c_void,
                            scratch_ptr,
                            k as i32,
                            k_padded as i32,
                            b_size as i32,
                            stream_ptr,
                        );
                    }
                    let launcher = plain_launcher_bf16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    if !packed_ready {
                        ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                            xs_ptr as *const std::ffi::c_void,
                            scratch_ptr,
                            k as i32,
                            k_padded as i32,
                            b_size as i32,
                            stream_ptr,
                        );
                    }
                    let launcher = plain_launcher_f16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    if !packed_ready {
                        ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                            xs_ptr as *const std::ffi::c_void,
                            scratch_ptr,
                            k as i32,
                            k_padded as i32,
                            b_size as i32,
                            stream_ptr,
                        );
                    }
                    let launcher = plain_launcher_f32(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

pub fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = gate_w.dtype();
    if dtype != up_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up dtype mismatch {:?} vs {:?}",
            dtype,
            up_w.dtype()
        );
    }
    let Some(launcher) = fused_glu_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_glu: unsupported dtype combination");
    };

    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("fast_mmvq fused_glu: gate/up weights are on different CUDA devices");
    }
    if !xs.device().same_device(&gate_w.device()) {
        candle_core::bail!("fast_mmvq fused_glu: input and weights are on different devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_glu: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_glu: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_glu: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_glu: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    // M38: this path quantises a different activation into the same workspace.
    workspace.invalidate();
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (gate_ptr, _gate_guard) = gate_w.device_ptr_with_guard(&stream)?;
    let (up_ptr, _up_guard) = up_w.device_ptr_with_guard(&stream)?;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let activation = activation as i32;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

/// Compute Q, K, and V matvecs with one input quantization pass and one MMVQ
/// kernel. The result tensors match the unfused `plain` outputs for each
/// projection and preserve the input dtype.
pub fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dtype = q_w.dtype();
    if dtype != k_w.dtype() || dtype != v_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v dtype mismatch {:?}, {:?}, {:?}",
            dtype,
            k_w.dtype(),
            v_w.dtype()
        );
    }
    let Some(launcher) = fused_qkv_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_qkv: unsupported dtype combination");
    };

    let Device::Cuda(dev) = q_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: q weight must live on CUDA");
    };
    let Device::Cuda(k_dev) = k_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: k weight must live on CUDA");
    };
    let Device::Cuda(v_dev) = v_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: v weight must live on CUDA");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        candle_core::bail!("fast_mmvq fused_qkv: q/k/v weights are on different CUDA devices");
    }
    if !xs.device().same_device(&q_w.device()) {
        candle_core::bail!("fast_mmvq fused_qkv: input and weights are on different devices");
    }

    let (q_nrows, ncols) = q_w.shape().dims2()?;
    let (k_nrows, k_ncols) = k_w.shape().dims2()?;
    let (v_nrows, v_ncols) = v_w.shape().dims2()?;
    if ncols != k_ncols || ncols != v_ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v ncols mismatch {ncols}, {k_ncols}, {v_ncols}"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_qkv: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: shape mismatch: weight ncols {ncols} vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_qkv: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_qkv: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_qkv: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    // M38: this path quantises a different activation into the same workspace.
    workspace.invalidate();
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut q_out = unsafe { dev.alloc::<half::bf16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::bf16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::bf16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut q_out = unsafe { dev.alloc::<half::f16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::f16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::f16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut q_out = unsafe { dev.alloc::<f32>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<f32>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<f32>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        _ => unreachable!(),
    }
}

/// M38: per-projection call counter for `plain`. Prints a histogram every 2 000 calls when
/// `NAIWA_MMVQ_DBG` is set, so the layer's GEMV mix can be read off one decode run.
mod mmvq_dbg {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use candle_core::quantized::GgmlDType;

    static ON: OnceLock<bool> = OnceLock::new();
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    fn on() -> bool {
        *ON.get_or_init(|| std::env::var_os("NAIWA_MMVQ_DBG").is_some())
    }

    fn table() -> &'static Mutex<HashMap<(GgmlDType, usize, usize, usize), usize>> {
        static T: OnceLock<Mutex<HashMap<(GgmlDType, usize, usize, usize), usize>>> = OnceLock::new();
        T.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub(super) fn record(dtype: GgmlDType, k: usize, nrows: usize, rows: usize) {
        if !on() {
            return;
        }
        let mut t = table().lock().unwrap();
        *t.entry((dtype, k, nrows, rows)).or_insert(0) += 1;
        let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 2000 != 0 {
            return;
        }
        let total: usize = t.values().sum();
        let mut rows_v: Vec<_> = t.iter().map(|(k, v)| (*k, *v)).collect();
        rows_v.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("MMVQ_DBG calls {total} (every 2000): dtype k nrows rows -> count");
        for ((d, k, nr, r), c) in rows_v.iter().take(24) {
            eprintln!(
                "MMVQ_DBG {d:?} k={k} nrows={nr} rows={r} -> {c} ({:.1}%)",
                100.0 * *c as f64 / total as f64
            );
        }
    }

    #[allow(dead_code)]
    static _UNUSED: AtomicBool = AtomicBool::new(false);
}
