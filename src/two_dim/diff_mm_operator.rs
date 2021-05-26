use super::G2D;

use ndarray::Array2;

pub fn construct_overlap_matrix_elements(gaussians: &Vec<G2D>) -> Array2<f64> {
    construct_diff_mm_matrix_elements((0, 0), (0, 0), (0.0, 0.0), gaussians)
}

pub fn construct_kinetic_operator_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    -0.5 * (construct_diff_mm_matrix_elements(
        (0, 0),
        (2, 0),
        (0.0, 0.0),
        gaussians,
    ) + construct_diff_mm_matrix_elements(
        (0, 0),
        (0, 2),
        (0.0, 0.0),
        gaussians,
    ))
}

pub fn construct_differential_operator_matrix_elements(
    f: (u32, u32),
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    construct_diff_mm_matrix_elements((0, 0), f, (0.0, 0.0), gaussians)
}

pub fn construct_multipole_moment_matrix_elements(
    e: (u32, u32),
    centers: (f64, f64),
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    construct_diff_mm_matrix_elements(e, (0, 0), centers, gaussians)
}

pub fn construct_angular_moment_z_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    // This function computes the matrix elements of angular moment operator in
    // z-direction multiplied by i to avoid having to deal with complex numbers.

    construct_diff_mm_matrix_elements((1, 0), (0, 1), (0.0, 0.0), gaussians)
        - construct_diff_mm_matrix_elements(
            (0, 1),
            (1, 0),
            (0.0, 0.0),
            gaussians,
        )
}

pub fn construct_diff_mm_matrix_elements(
    e: (u32, u32),
    f: (u32, u32),
    centers: (f64, f64),
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    let gaussians_x: Vec<crate::one_dim::G1D> =
        gaussians.iter().map(|g| g.g_x).collect();
    let gaussians_y: Vec<crate::one_dim::G1D> =
        gaussians.iter().map(|g| g.g_y).collect();

    crate::one_dim::construct_diff_mm_matrix_elements(
        e.0,
        f.0,
        centers.0,
        &gaussians_x,
    ) * crate::one_dim::construct_diff_mm_matrix_elements(
        e.1,
        f.1,
        centers.1,
        &gaussians_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    #[test]
    fn test_overlap_matrix_elements() {
        let omega = 1.0;

        let gaussians = vec![
            G2D::new((0, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 2), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((2, 0), omega / 2.0, (0.0, 0.0)),
        ];

        let s = construct_overlap_matrix_elements(&gaussians);
        assert!(s.is_square());

        for i in 0..s.nrows() {
            assert_abs_diff_eq!(s[[i, i]], 1.0, epsilon = 1e-12);
            for j in 0..s.ncols() {
                assert_abs_diff_eq!(s[[i, j]], s[[j, i]]);
            }
        }
    }

    #[test]
    fn test_dipole_matrix_elements() {
        let omega = 0.5;

        let gaussians = vec![
            G2D::new((0, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 2), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((2, 0), omega / 2.0, (0.0, 0.0)),
        ];

        let d_x = construct_multipole_moment_matrix_elements(
            (1, 0),
            (0.0, 0.0),
            &gaussians,
        );
        assert!(d_x.is_square());

        let d_y = construct_multipole_moment_matrix_elements(
            (0, 1),
            (0.0, 0.0),
            &gaussians,
        );
        assert!(d_y.is_square());
        assert!(d_x.nrows() == d_y.nrows());

        for i in 0..d_x.nrows() {
            for j in 0..d_x.ncols() {
                assert_abs_diff_eq!(d_x[[i, j]], d_x[[j, i]]);
                assert_abs_diff_eq!(d_y[[i, j]], d_y[[j, i]]);
            }
        }

        assert_abs_diff_eq!(d_x[[0, 0]], d_y[[0, 0]]);
        assert_abs_diff_eq!(d_x[[0, 1]], d_y[[0, 2]]);
        assert_abs_diff_eq!(d_x[[0, 2]], d_y[[0, 1]]);
        assert_abs_diff_eq!(d_x[[0, 3]], d_y[[0, 5]]);
        assert_abs_diff_eq!(d_x[[0, 4]], d_y[[0, 4]]);
        assert_abs_diff_eq!(d_x[[0, 5]], d_y[[0, 3]]);
    }

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

    fn construct_angular_moment_z_matrix_elements(
        gaussians: &Vec<G2D>,
    ) -> Array2<f64> {
        // This function computes the matrix elements of angular moment operator in
        // z-direction multiplied by i to avoid having to deal with complex numbers.

        let gaussians_x: Vec<crate::one_dim::G1D> =
            gaussians.iter().map(|g| g.g_x).collect();
        let gaussians_y: Vec<crate::one_dim::G1D> =
            gaussians.iter().map(|g| g.g_y).collect();

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

    #[test]
    fn test_angular_moment() {
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

        let ilz = super::construct_angular_moment_z_matrix_elements(&gaussians);
        let ilz_2 =
            tests::construct_angular_moment_z_matrix_elements(&gaussians);

        for p in 0..l {
            for q in 0..l {
                assert_abs_diff_eq!(ilz[[p, q]], ilz[[q, p]]);
                assert_abs_diff_eq!(ilz[[p, q]], ilz_2[[p, q]]);
            }
        }
    }
}
