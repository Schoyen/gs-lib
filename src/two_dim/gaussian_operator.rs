use super::G2D;

use ndarray::Array2;

pub fn construct_gaussian_operator_matrix_elements(
    op: &G2D,
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    let op_x = op.g_x;
    let op_y = op.g_y;

    let gaussians_x: Vec<crate::one_dim::G1D> =
        gaussians.iter().map(|g| g.g_x).collect();
    let gaussians_y: Vec<crate::one_dim::G1D> =
        gaussians.iter().map(|g| g.g_y).collect();

    crate::one_dim::construct_gaussian_operator_matrix_elements(
        &op_x,
        &gaussians_x,
    ) * crate::one_dim::construct_gaussian_operator_matrix_elements(
        &op_y,
        &gaussians_y,
    )
}
