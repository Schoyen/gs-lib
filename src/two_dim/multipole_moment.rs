use super::G2D;
use crate::one_dim::G1D;

use ndarray::Array2;

pub fn construct_overlap_matrix_elements(gaussians: &Vec<G2D>) -> Array2<f64> {
    construct_multipole_moment_matrix_elements((0, 0), (0.0, 0.0), gaussians)
}

pub fn construct_multipole_moment_matrix_elements(
    e: (u32, u32),
    centers: (f64, f64),
    gaussians: &Vec<G2D>,
) -> Array2<f64> {
    // Note: This creates copies of every G1D
    let gaussians_x: Vec<G1D> = gaussians.iter().map(|g| g.g_x).collect();
    let gaussians_y: Vec<G1D> = gaussians.iter().map(|g| g.g_y).collect();

    crate::one_dim::construct_multipole_moment_matrix_elements(
        e.0,
        centers.0,
        &gaussians_x,
    ) * crate::one_dim::construct_multipole_moment_matrix_elements(
        e.1,
        centers.1,
        &gaussians_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
}
