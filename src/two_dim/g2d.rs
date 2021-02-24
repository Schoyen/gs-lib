use crate::one_dim::G1D;
use ndarray::{Array, Array2};

#[derive(Debug)]
pub struct G2D {
    pub g_x: G1D,
    pub g_y: G1D,
    pub norm: f64,
}

impl G2D {
    pub fn new(alpha: (u32, u32), a: f64, centers: (f64, f64)) -> Self {
        let g_x = G1D::new(alpha.0, a, centers.0, 'x');
        let g_y = G1D::new(alpha.1, a, centers.1, 'y');

        let norm = g_x.norm * g_y.norm;

        G2D { g_x, g_y, norm }
    }

    pub fn evaluate_point(&self, x: f64, y: f64, with_norm: bool) -> f64 {
        self.g_x.evaluate_point(x, with_norm)
            * self.g_y.evaluate_point(y, with_norm)
    }

    pub fn evaluate(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        with_norm: bool,
    ) -> Array2<f64> {
        assert!(x.nrows() == y.nrows());
        assert!(x.ncols() == y.ncols());

        let mut eval = Array::zeros((x.nrows(), x.ncols()));

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                eval[[i, j]] +=
                    self.evaluate_point(x[[i, j]], y[[i, j]], with_norm);
            }
        }

        eval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_construction() {
        let omega = 1.0;

        let gaussians = vec![
            G2D::new((0, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 2), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((2, 0), omega / 2.0, (0.0, 0.0)),
        ];

        for gauss in gaussians.iter() {
            assert_abs_diff_eq!(gauss.norm, gauss.g_x.norm * gauss.g_y.norm);
        }
    }
}
