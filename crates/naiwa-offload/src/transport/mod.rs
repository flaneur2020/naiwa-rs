//! Layer 1 (`transport`): hardware asynchronous primitives (CUDA copy streams, mapped doorbells,
//! advisory activation channels, single-thread CPU FIFO executor, and spin-wait parallel pool).
//!
//! Strictly depends only on cross-cutting `config` / `stats`; contains no MoE cache or pipeline logic.

pub(crate) mod activation_channel;
pub(crate) mod copy_stream;
pub(crate) mod doorbell;
pub(crate) mod executor;
pub mod spin_pool;

pub(crate) use activation_channel::LookaheadChannel;
pub(crate) use copy_stream::{CopyDone, CopyStream, CopyStreams};
pub(crate) use doorbell::{PassCell, TopkDoorbell};
