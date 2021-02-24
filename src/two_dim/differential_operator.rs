use super::G2D;
use crate::one_dim::G1D;

use ndarray::Array2;

pub fn construct_kinetic_operator_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    -0.5 * (construct_differential_operator_matrix_elements((2, 0), gaussians)
        + construct_differential_operator_matrix_elements((0, 2), gaussians))
}

pub fn construct_differential_operator_matrix_elements(
    e: (u32, u32),
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    // Note: This creates copies of every G1D
    let gaussians_x: Vec<G1D> = gaussians.iter().map(|g| g.g_x).collect();
    let gaussians_y: Vec<G1D> = gaussians.iter().map(|g| g.g_y).collect();

    crate::one_dim::construct_differential_operator_matrix_elements(
        e.0,
        &gaussians_x,
    ) * crate::one_dim::construct_differential_operator_matrix_elements(
        e.1,
        &gaussians_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kinetic_energy_operator() {
        let omega = 0.5;

        let gaussians = vec![
            G2D::new((0, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 2), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((2, 0), omega / 2.0, (0.0, 0.0)),
        ];

        let t = construct_kinetic_operator_matrix_elements(&gaussians);
        assert!(t.is_square());

        for i in 0..t.nrows() {
            for j in 0..t.ncols() {
                assert_abs_diff_eq!(t[[i, j]], t[[j, i]]);
            }
        }
    }
}
