//! Decode components: route, gpu, cpu-miss, dma-miss and preload stages.
//!
//! Each component has one entry and one exit; the orchestrator lives in `crate::decode`.

mod cpu_miss;
mod dma_miss;
mod gpu;
mod preload;
mod route;

pub(crate) use cpu_miss::submit as submit_cpu_miss;
pub(crate) use dma_miss::stage as stage_dma_miss;
pub(crate) use gpu::run as run_gpu;
pub(crate) use preload::submit as submit_preload;
pub(crate) use route::split as split_route;
pub(crate) use route::RoutePlan;

pub(crate) type Hit = (std::sync::Arc<crate::moe::SlotBuf>, usize, f32);
