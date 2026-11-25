//! Shared plumbing for every VIctory crate and tool.
//!
//! * [`config`]: where the user's files live (never inside the repository).
//! * [`policy`]: the runtime half of rule #1, refusing to write
//!   game-derived data into the source tree.
//! * [`logging`]: one tracing setup for all binaries.
//! * [`parallel`]: work spread over threads where the result cannot depend
//!   on how (a BLAKE3 hash equal to the serial one, independent items).

pub mod config;
pub mod logging;
pub mod parallel;
pub mod policy;

pub use config::Config;
pub use policy::{PolicyError, ensure_outside_repo};
