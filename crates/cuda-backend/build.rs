// Links cuBLAS from the CUDA toolkit when the `cuda` feature is on.

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit =
        std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    println!("cargo:rustc-link-search=native={toolkit}/lib64");
    println!("cargo:rustc-link-lib=dylib=cublas");
}
