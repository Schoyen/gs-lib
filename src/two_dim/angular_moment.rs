use super::G2D;
use crate::one_dim::G1D;

use ndarray::Array2;

pub fn construct_angular_moment_z_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    // This function computes the matrix elements of angular moment operator in
    // z-direction multiplies by i to avoid having to deal with complex numbers.

    let gaussians_x: Vec<G1D> = gaussians.iter().map(|g| g.g_x).collect();
    let gaussians_y: Vec<G1D> = gaussians.iter().map(|g| g.g_y).collect();

    let x_dy = crate::one_dim::construct_multipole_moment_matrix_elements(
        1,
        0.0,
        &gaussians_x,
    )
        * crate::one_dim::construct_differential_operator_matrix_elements(
            1,
            &gaussians_y,
        );

    let y_dx = crate::one_dim::construct_multipole_moment_matrix_elements(
        1,
        0.0,
        &gaussians_y,
    )
        * crate::one_dim::construct_differential_operator_matrix_elements(
            1,
            &gaussians_x,
        );

    x_dy - y_dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    #[test]
    fn test_symmetry() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..5 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..2), rng.gen_range(0..2)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        let l = gaussians.len();

        let ilz = construct_angular_moment_z_matrix_elements(&gaussians);

        for p in 0..l {
            for q in 0..l {
                assert_abs_diff_eq!(ilz[[p, q]], ilz[[q, p]]);
            }
        }
    }
}
