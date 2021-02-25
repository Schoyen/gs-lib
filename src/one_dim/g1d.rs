use crate::math::double_factorial;
use ndarray::Array1;

#[derive(Debug, Copy, Clone)]
pub struct G1D {
    pub i: u32,
    pub a: f64,
    pub center: f64,
    pub symbol: char,
    pub norm: f64,
}

impl G1D {
    pub fn new(i: u32, a: f64, center: f64, symbol: char) -> Self {
        let norm = 1.0 / G1D::compute_norm(i, a);

        G1D {
            i,
            a,
            center,
            symbol,
            norm,
        }
    }

    pub fn compute_norm(i: u32, a: f64) -> f64 {
        let df_test = 2 * (i as i32) - 1;
        assert!(df_test >= -1);

        let df = if df_test == -1 {
            1.0
        } else {
            double_factorial(df_test as i64) as f64
        };

        (df / (4.0 * a).powi(i as i32)
            * (std::f64::consts::PI / (2.0 * a)).sqrt())
        .sqrt()
    }

    pub fn evaluate_point(&self, x: f64, with_norm: bool) -> f64 {
        let norm = if with_norm { self.norm } else { 1.0 };
        let x_center = x - self.center;

        norm * x_center.powi(self.i as i32) * (-self.a * x_center.powi(2)).exp()
    }

    pub fn evaluate(&self, x: &Array1<f64>, with_norm: bool) -> Array1<f64> {
        x.mapv(|x| self.evaluate_point(x, with_norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_construction() {
        let g1d_z = G1D::new(0, 1.0, 0.0, 'x');

        assert!(G1D::compute_norm(1, 1.0) > 0.0);
        assert!((G1D::compute_norm(0, 1.0) - 1.11951).abs() < 1e-3);
        assert_eq!(g1d_z.norm, 1.0 / G1D::compute_norm(0, 1.0));
    }

    #[test]
    fn test_recurrence() {
        let x = Array::linspace(-2.0, 2.0, 101);

        let a = 1.0;
        let center = 0.5;
        let mut g_c = G1D::new(0, a, center, 'x');

        for i in 1..4 {
            let g = G1D::new(i, a, center, 'x');
            let g_left = g.evaluate(&x, false);
            let g_right = g_c.evaluate(&x, false) * (&x - center);

            for (left, right) in g_left.iter().zip(g_right.iter()) {
                assert_abs_diff_eq!(left, right);
            }

            g_c = g;
        }
    }
}
