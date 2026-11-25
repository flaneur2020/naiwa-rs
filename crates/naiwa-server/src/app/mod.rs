//! Application layer: CLI parsing, local GGUF model loading, Axum HTTP API, and server lifecycle.

pub mod args;
pub mod http;
pub mod load;
pub mod serve;

pub use args::{Cli, Command};
pub use serve::run_serve;
