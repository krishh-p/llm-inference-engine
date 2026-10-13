use std::process::ExitCode;

// The committed fixtures, located from this crate so the working directory does
// not matter. A different directory can be passed as the first argument.
const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures");

fn main() -> ExitCode {
    let fixture_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| FIXTURE_DIR.to_owned());

    match cuda_backend::run_smoke(fixture_dir) {
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
