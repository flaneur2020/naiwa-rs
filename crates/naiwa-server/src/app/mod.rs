//! Layer 4 (`app`): CLI arguments, local model loader, and HTTP server wiring.

pub(crate) mod args;
pub(crate) mod load;
pub(crate) mod mistralrs_bridge;
pub(crate) mod serve;

pub(crate) use args::{Cli, Command};
pub(crate) use serve::run_serve;
