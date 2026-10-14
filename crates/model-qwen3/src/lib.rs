mod artifacts;
mod config;
mod error;
mod generate;
mod prompts;
mod tokenizer;
mod verify;
mod weights;

#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod block;
mod kv_cache;
#[cfg(feature = "cuda")]
mod mlp;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod trace;
#[cfg(feature = "cuda")]
mod workspace;

pub use artifacts::{ModelSummary, Qwen3Artifacts};
pub use config::Qwen3Config;
pub use error::{ModelError, Result};
pub use generate::{Completion, complete};
pub use prompts::{RegressionCase, RegressionPromptSet};
pub use tokenizer::Qwen3Tokenizer;
pub use verify::{
    BoundaryResult, GenerationResult, verify_generation, verify_layer_0, verify_model,
};
