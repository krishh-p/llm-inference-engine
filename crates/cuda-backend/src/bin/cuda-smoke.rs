use std::process::ExitCode;

// The committed fixture, located from this crate so the working directory does
// not matter. A different file can be passed as the first argument.
const RMS_NORM_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/rms_norm.safetensors"
);

fn main() -> ExitCode {
    let fixture = std::env::args()
        .nth(1)
        .unwrap_or_else(|| RMS_NORM_FIXTURE.to_owned());

    match cuda_backend::run_smoke(fixture) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: could not serialize smoke report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
