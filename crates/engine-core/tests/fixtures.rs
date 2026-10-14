use std::path::PathBuf;

use engine_core::{DType, FixtureManifest, SafeTensorCatalog};

// The tensors the Rust checks read from each fixture file.
const REQUIRED_TENSORS: [(&str, &[&str]); 5] = [
    (
        "block_0.safetensors",
        &[
            "input",
            "input_norm",
            "q",
            "k",
            "v",
            "q_rope",
            "k_rope",
            "scores",
            "attention_probs",
            "attention_output",
            "mlp_input",
            "mlp_output",
            "output",
        ],
    ),
    (
        "decode_block_0.safetensors",
        &[
            "input",
            "input_norm",
            "q",
            "k",
            "v",
            "q_rope",
            "k_rope",
            "cache_k_in",
            "cache_v_in",
            "cache_k_out",
            "cache_v_out",
            "scores",
            "attention_probs",
            "attention_output",
            "mlp_input",
            "mlp_output",
            "output",
        ],
    ),
    ("generation.safetensors", &["final_norm", "prefill_logits"]),
    (
        "rms_norm.safetensors",
        &[
            "input",
            "weight",
            "eps",
            "output",
            "q_input",
            "q_weight",
            "q_output",
            "bf16_input",
            "bf16_weight",
            "bf16_output",
            "wide_input",
            "wide_weight",
            "wide_output",
        ],
    ),
    (
        "rope.safetensors",
        &[
            "q_input",
            "k_input",
            "positions",
            "theta",
            "cos",
            "sin",
            "q_output",
            "k_output",
        ],
    ),
];

fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

#[test]
fn committed_fixture_manifest_matches_its_files() {
    let directory = fixture_directory();
    let manifest = FixtureManifest::load(directory.join("manifest.json")).unwrap();

    // checks every file's size, checksum and tensor metadata
    let validation = manifest.validate_files(&directory).unwrap();

    assert_eq!(validation.file_count, REQUIRED_TENSORS.len());
}

#[test]
fn manifest_lists_every_required_tensor() {
    let manifest = FixtureManifest::load(fixture_directory().join("manifest.json")).unwrap();

    for (file, tensors) in REQUIRED_TENSORS {
        let recorded = &manifest
            .files
            .get(file)
            .unwrap_or_else(|| panic!("{file} is missing from the manifest"))
            .tensors;
        for tensor in tensors {
            assert!(recorded.contains_key(*tensor), "{file} has no {tensor}");
        }
    }
}

#[test]
fn fixtures_were_generated_from_the_pinned_revision() {
    let manifest = FixtureManifest::load(fixture_directory().join("manifest.json")).unwrap();
    let pinned = include_str!("../../../model_revision.txt").trim();

    assert_eq!(manifest.model.revision, pinned);
}

#[test]
fn loads_a_selected_fixture_tensor() {
    let path = fixture_directory().join("rms_norm.safetensors");
    let catalog = SafeTensorCatalog::open_files([path]).unwrap();
    let tensor = catalog.load("eps").unwrap();

    assert_eq!(tensor.descriptor.layout.dtype, DType::Float32);
    assert_eq!(tensor.descriptor.layout.shape, Vec::<usize>::new());
    assert_eq!(tensor.bytes.len(), 4);
}

#[test]
fn missing_tensor_error_names_the_tensor() {
    let path = fixture_directory().join("rms_norm.safetensors");
    let catalog = SafeTensorCatalog::open_files([path]).unwrap();
    let error = catalog.load("not_a_tensor").unwrap_err();

    assert_eq!(error.to_string(), "tensor \"not_a_tensor\" was not found");
}
