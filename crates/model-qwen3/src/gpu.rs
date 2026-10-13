use cuda_backend::Backend;

use crate::projections::{Projection, compare, split_heads};
use crate::{ProjectionResult, Qwen3Config, Result};

/// Runs every projection on its fixture inputs and checks the outputs.
pub(crate) fn run_projections(
    config: &Qwen3Config,
    projections: &[Projection],
) -> Result<Vec<ProjectionResult>> {
    let backend = Backend::new()?;
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let mut results = Vec::new();

    for projection in projections {
        let heads = projection.heads;
        let width = heads * head_dim;
        let weight = backend.upload_matrix(&projection.weight, [width, hidden_size])?;

        for case in &projection.cases {
            let rows = case.rows;
            let input = backend.upload_matrix(&case.input, [rows, hidden_size])?;
            let mut output = backend.zeros_matrix([rows, width])?;

            // the reference computes this as `x @ weight.T`
            backend.matmul_transposed(&mut output, &input, &weight)?;
            let actual = split_heads(&backend.download(output)?, rows, heads, head_dim);

            let name = format!("{}_{}", case.name, projection.name);
            let max_abs_error = compare(&name, &actual, &case.expected, case.tolerance)?;
            results.push(ProjectionResult {
                name,
                shape: [heads, rows, head_dim],
                max_abs_error,
                tolerance: case.tolerance,
            });
        }
    }

    Ok(results)
}
