use std::path::PathBuf;

use engine_core::{DType, FixtureManifest, SafeTensorCatalog};

fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

#[test]
fn committed_fixture_manifest_matches_its_files() {
    let directory = fixture_directory();
    let manifest = FixtureManifest::load(directory.join("manifest.json")).unwrap();
    let validation = manifest.validate_files(&directory).unwrap();

    assert_eq!(validation.file_count, 5);
    assert_eq!(validation.tensor_count, 51);
    assert_eq!(validation.total_bytes, 1_571_048);
    assert_eq!(
        manifest
            .files
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "block_0.safetensors",
            "decode_block_0.safetensors",
            "generation.safetensors",
            "rms_norm.safetensors",
            "rope.safetensors",
        ]
    );
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
