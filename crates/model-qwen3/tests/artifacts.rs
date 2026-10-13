use std::collections::HashMap;
use std::fs;
use std::path::Path;

use model_qwen3::{ModelError, Qwen3Artifacts, Qwen3Tokenizer, RegressionPromptSet};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use serde_json::json;
use tempfile::TempDir;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{SplitDelimiterBehavior, Tokenizer};

const HIDDEN_SIZE: usize = 8;
const INTERMEDIATE_SIZE: usize = 12;
const HEAD_DIM: usize = 4;
const VOCAB_SIZE: usize = 16;

#[test]
fn opens_and_summarizes_a_valid_snapshot() {
    let directory = TempDir::new().unwrap();
    write_snapshot(directory.path(), None);

    let artifacts = Qwen3Artifacts::open(directory.path()).unwrap();
    let summary = artifacts.summary();

    assert_eq!(summary.architecture, "Qwen3ForCausalLM");
    assert_eq!(summary.hidden_size, HIDDEN_SIZE);
    assert_eq!(summary.intermediate_size, INTERMEDIATE_SIZE);
    assert_eq!(summary.layers, 1);
    assert_eq!(summary.attention_heads, 2);
    assert_eq!(summary.key_value_heads, 1);
    assert_eq!(summary.head_dim, HEAD_DIM);
    assert_eq!(summary.vocabulary_size, VOCAB_SIZE);
    assert_eq!(summary.max_sequence_length, 32);
    assert_eq!(summary.weight_files, 1);
    assert_eq!(summary.tensors, 13);
    assert_eq!(summary.stored_parameters, 640);
    assert_eq!(summary.dtypes, vec!["bfloat16"]);
}

#[test]
fn rejects_a_weight_with_the_wrong_shape() {
    let directory = TempDir::new().unwrap();
    write_snapshot(
        directory.path(),
        Some(("model.layers.0.self_attn.q_proj.weight", vec![7, 8])),
    );

    let error = Qwen3Artifacts::open(directory.path()).err().unwrap();
    assert!(matches!(error, ModelError::ShapeMismatch { .. }));
    assert_eq!(
        error.to_string(),
        "tensor \"model.layers.0.self_attn.q_proj.weight\" has shape [7, 8], expected [8, 8]"
    );
}

#[test]
fn missing_snapshot_reports_the_required_file() {
    let directory = TempDir::new().unwrap();
    let expected = directory.path().join("config.json");

    let error = Qwen3Artifacts::open(directory.path()).err().unwrap();
    assert!(matches!(error, ModelError::MissingArtifact(_)));
    assert_eq!(
        error.to_string(),
        format!("required model artifact is missing: {}", expected.display())
    );
}

#[test]
fn regression_set_accepts_a_tokenizer_built_from_its_ids() {
    let directory = TempDir::new().unwrap();
    let tokenizer_path = directory.path().join("tokenizer.json");
    write_regression_tokenizer(&tokenizer_path);
    let tokenizer = Qwen3Tokenizer::load(&tokenizer_path).unwrap();

    regression_prompts().validate_tokenizer(&tokenizer).unwrap();
}

#[test]
#[ignore = "needs MODEL_DIR set to a local Qwen3 snapshot"]
fn qwen3_tokenizer_matches_recorded_prompt_ids() {
    let model_dir = std::env::var("MODEL_DIR").expect("MODEL_DIR is not set");
    let tokenizer = Qwen3Tokenizer::load(Path::new(&model_dir).join("tokenizer.json")).unwrap();

    regression_prompts().validate_tokenizer(&tokenizer).unwrap();
}

fn regression_prompts() -> RegressionPromptSet {
    RegressionPromptSet::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/prompts.json"),
    )
    .unwrap()
}

fn write_snapshot(directory: &Path, shape_override: Option<(&str, Vec<usize>)>) {
    fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&json!({
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "head_dim": HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_position_embeddings": 32,
            "model_type": "qwen3",
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "vocab_size": VOCAB_SIZE
        }))
        .unwrap(),
    )
    .unwrap();
    write_minimal_tokenizer(&directory.join("tokenizer.json"));

    let mut tensors = vec![
        ("model.embed_tokens.weight", vec![VOCAB_SIZE, HIDDEN_SIZE]),
        ("model.norm.weight", vec![HIDDEN_SIZE]),
        ("model.layers.0.input_layernorm.weight", vec![HIDDEN_SIZE]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2 * HEAD_DIM, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![HEAD_DIM, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![HEAD_DIM, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![HIDDEN_SIZE, 2 * HEAD_DIM],
        ),
        ("model.layers.0.self_attn.q_norm.weight", vec![HEAD_DIM]),
        ("model.layers.0.self_attn.k_norm.weight", vec![HEAD_DIM]),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
        ),
    ];
    if let Some((name, shape)) = shape_override {
        tensors
            .iter_mut()
            .find(|(tensor_name, _)| *tensor_name == name)
            .unwrap()
            .1 = shape;
    }

    let owned = tensors
        .into_iter()
        .map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * 2];
            (name.to_owned(), shape, bytes)
        })
        .collect::<Vec<_>>();
    let views = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name,
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.join("model.safetensors")).unwrap();
}

fn write_minimal_tokenizer(path: &Path) {
    let vocab = [("<unk>".to_owned(), 0)]
        .into_iter()
        .collect::<HashMap<_, _>>();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_owned())
        .build()
        .unwrap();
    Tokenizer::new(model).save(path, false).unwrap();
}

fn write_regression_tokenizer(path: &Path) {
    let mut vocab = (0..=75_698)
        .map(|id| (format!("<unused_{id}>"), id))
        .collect::<HashMap<_, _>>();
    for (token, id) in [
        ("<unk>", 0),
        ("The", 785),
        ("Ġcapital", 6722),
        ("Ġof", 315),
        ("ĠFrance", 9625),
        ("Ġis", 374),
        ("2", 17),
        ("Ġ+", 488),
        ("Ġ", 220),
        ("Ġ=", 284),
        ("Once", 12_522),
        ("Ġupon", 5193),
        ("Ġa", 264),
        ("Ġtime", 882),
        ("def", 750),
        ("Ġfibonacci", 75_698),
        ("(n", 1445),
        ("):", 1648),
        ("Ġopposite", 14_002),
        ("Ġhot", 4017),
    ] {
        vocab.remove(&format!("<unused_{id}>"));
        vocab.insert(token.to_owned(), id);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    let split = Split::new(
        SplitPattern::Regex(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                .to_owned(),
        ),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap();
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ])));
    tokenizer.save(path, false).unwrap();
}
