//! Device-to-host doorbells: the device publishes, the host waits and reads.

mod activation;
mod topk;

pub use activation::ActivationDoorbell;
pub use topk::TopkDoorbell;
