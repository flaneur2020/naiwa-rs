//! Decode stages: route, gpu, cpu-miss, dma-miss and preload.
//!
//! Each stage is a standalone struct with one `submit` entry and one exit; the orchestrator
//! lives in `super::decode`.

mod cpu_miss;
mod dma_miss;
mod gpu;
mod preload;
mod route;

pub(crate) use cpu_miss::{fused_decode, CpuMiss};
pub(crate) use dma_miss::DmaMiss;
pub(crate) use gpu::Gpu;
pub(crate) use preload::Preload;
pub(crate) use route::{Route, RoutePlan};

pub(crate) type Hit = (std::sync::Arc<crate::moe::SlotBuf>, usize, f32);
