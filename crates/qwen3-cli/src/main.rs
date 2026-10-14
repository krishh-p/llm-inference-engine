use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use engine_core::FixtureManifest;
use model_qwen3::{
    BoundaryResult, ModelError, Qwen3Artifacts, RegressionPromptSet, verify_generation,
    verify_layer_0, verify_model,
};
use thiserror::Error;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-1.7B-Base";

#[derive(Debug, Parser)]
#[command(name = "qwen3-cli", about = "Inspect and verify Qwen3 model artifacts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a local Qwen3 snapshot and the committed golden fixtures.
    Inspect {
        /// Directory containing config.json, tokenizer.json and `SafeTensors` weights.
        #[arg(long)]
        model_dir: PathBuf,

        /// Directory containing manifest.json and the golden `SafeTensors` files.
        #[arg(long, default_value = "tests/fixtures")]
        fixtures: PathBuf,

        /// JSON file containing fixed prompts and their expected token IDs.
        #[arg(long, default_value = "tests/prompts.json")]
        prompts: PathBuf,

        /// File containing the pinned Hugging Face model revision.
        #[arg(long, default_value = "model_revision.txt")]
        revision_file: PathBuf,

        /// Hugging Face model ID expected by the fixture manifest.
        #[arg(long, default_value = DEFAULT_MODEL_ID)]
        model_id: String,
    },

    /// Run layer 0 and the whole model on the GPU, compare them with the golden
    /// fixtures, and check the greedy tokens for every regression prompt.
    Verify {
        /// Directory containing config.json, tokenizer.json and `SafeTensors` weights.
        #[arg(long)]
        model_dir: PathBuf,

        /// Directory containing manifest.json and the golden `SafeTensors` files.
        #[arg(long, default_value = "tests/fixtures")]
        fixtures: PathBuf,

        /// JSON file containing fixed prompts and the tokens they should generate.
        #[arg(long, default_value = "tests/prompts.json")]
        prompts: PathBuf,
    },
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Engine(#[from] engine_core::EngineError),

    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("fixture model ID {actual:?} does not match expected model ID {expected:?}")]
    ModelId { expected: String, actual: String },

    #[error(
        "fixture prompt tokens do not match tokenizer output: expected {expected:?}, got {actual:?}"
    )]
    FixtureTokens {
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Inspect {
            model_dir,
            fixtures,
            prompts,
            revision_file,
            model_id,
        } => inspect(&model_dir, &fixtures, &prompts, &revision_file, &model_id),
        Command::Verify {
            model_dir,
            fixtures,
            prompts,
        } => verify(&model_dir, &fixtures, &prompts),
    }
}

fn inspect(
    model_dir: &Path,
    fixture_dir: &Path,
    prompts_path: &Path,
    revision_path: &Path,
    expected_model_id: &str,
) -> Result<(), CliError> {
    let pinned_revision = read_trimmed(revision_path)?;
    let artifacts = Qwen3Artifacts::open(model_dir)?;
    let manifest = FixtureManifest::load(fixture_dir.join("manifest.json"))?;

    if manifest.model.id != expected_model_id {
        return Err(CliError::ModelId {
            expected: expected_model_id.to_owned(),
            actual: manifest.model.id,
        });
    }
    if manifest.model.revision != pinned_revision {
        return Err(ModelError::RevisionMismatch {
            expected: pinned_revision,
            actual: manifest.model.revision,
        }
        .into());
    }

    let fixture_validation = manifest.validate_files(fixture_dir)?;
    let prompts = RegressionPromptSet::load(prompts_path)?;
    prompts.validate_tokenizer(&artifacts.tokenizer)?;
    let fixture_token_ids = artifacts.tokenizer.encode(&manifest.prompt)?;
    if fixture_token_ids != manifest.token_ids {
        return Err(CliError::FixtureTokens {
            expected: manifest.token_ids,
            actual: fixture_token_ids,
        });
    }

    let summary = artifacts.summary();
    println!("model: {expected_model_id}");
    println!("revision: {}", manifest.model.revision);
    println!("architecture: {}", summary.architecture);
    println!("fixture_dtype: {}", manifest.model.dtype);
    println!("layers: {}", summary.layers);
    println!("hidden_size: {}", summary.hidden_size);
    println!("intermediate_size: {}", summary.intermediate_size);
    println!("attention_heads: {}", summary.attention_heads);
    println!("key_value_heads: {}", summary.key_value_heads);
    println!("head_dim: {}", summary.head_dim);
    println!("vocabulary_size: {}", summary.vocabulary_size);
    println!("max_sequence_length: {}", summary.max_sequence_length);
    println!("weight_files: {}", summary.weight_files);
    println!("weight_tensors: {}", summary.tensors);
    println!("stored_parameters: {}", summary.stored_parameters);
    println!("weight_dtypes: {}", summary.dtypes.join(", "));
    println!("regression_prompts: {}", prompts.cases.len());
    println!("fixture_files: {}", fixture_validation.file_count);
    println!("fixture_tensors: {}", fixture_validation.tensor_count);
    println!("fixture_bytes: {}", fixture_validation.total_bytes);
    println!("artifact validation passed");
    Ok(())
}

fn verify(model_dir: &Path, fixture_dir: &Path, prompts_path: &Path) -> Result<(), CliError> {
    let artifacts = Qwen3Artifacts::open(model_dir)?;

    let results = verify_layer_0(&artifacts, fixture_dir)?;
    print_results(&results);
    println!("layer 0 checks passed: {}", results.len());

    let results = verify_model(&artifacts, fixture_dir)?;
    print_results(&results);
    println!("model checks passed: {}", results.len());

    let prompts = RegressionPromptSet::load(prompts_path)?;
    let results = verify_generation(&artifacts, &prompts)?;
    for result in &results {
        println!(
            "{:?} (stops at {:?}): {:?} {:?}",
            result.prompt,
            result.eos_ids,
            result.token_ids,
            artifacts.tokenizer.decode(&result.token_ids)?
        );
    }
    println!(
        "generation checks passed, with and without caches: {}",
        results.len()
    );
    Ok(())
}

fn print_results(results: &[BoundaryResult]) {
    for result in results {
        println!(
            "{}: max_abs_error={}, atol={}, rtol={}",
            result.name, result.max_abs_error, result.tolerance.atol, result.tolerance.rtol
        );
    }
}

fn read_trimmed(path: &Path) -> Result<String, CliError> {
    std::fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(|source| CliError::Io {
            path: path.to_path_buf(),
            source,
        })
}
