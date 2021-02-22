use factorial::DoubleFactorial;
use itertools::izip;

pub struct G1D {
    pub i: u32,
    pub a: f64,
    pub center: f64,
    pub symbol: char,
    pub norm: f64,
}

impl G1D {
    pub fn new(i: u32, a: f64, center: f64, symbol: char) -> Self {
        let norm = G1D::compute_norm(i, a);

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
            1
        } else {
            (df_test as u32).double_factorial()
        };

        ((df as f64) / (4.0 * a).powi(i as i32)
            * (std::f64::consts::PI / (2.0 * a)).sqrt())
        .sqrt()
    }

    pub fn evaluate_point(&self, x: f64, with_norm: bool) -> f64 {
        let norm = if with_norm { self.norm } else { 1.0 };
        let x_center = x - self.center;

        norm * x_center.powi(self.i as i32) * (-self.a * x_center.powi(2)).exp()
    }

    pub fn evaluate(&self, x: &Vec<f64>, with_norm: bool) -> Vec<f64> {
        let mut res = vec![0.0; x.len()];

        for i in 0..x.len() {
            res[i] = self.evaluate_point(x[i], with_norm);
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let g1d_z = G1D::new(0, 1.0, 0.0, 'x');

        assert!(G1D::compute_norm(1, 1.0) > 0.0);
        assert!((G1D::compute_norm(0, 1.0) - 1.11951).abs() < 1e-3);
        assert_eq!(g1d_z.norm, G1D::compute_norm(0, 1.0));
    }

    #[test]
    fn test_recurrence() {
        let start = -2.0;
        let stop = 2.0;
        let n = 101;
        let dx = (stop - start) / ((n - 1) as f64);
        let mut x = vec![0.0; n];

        for i in 0..n {
            x[i] = start + (i as f64) * dx;
        }

        assert_eq!(x[x.len() - 1], stop);

        let a = 1.0;
        let center = 0.5;
        let mut g_c = G1D::new(0, a, center, 'x');

        for i in 1..4 {
            let g = G1D::new(i, a, center, 'x');

            for (_x, y_c, y) in
                izip!(&x, g_c.evaluate(&x, false), g.evaluate(&x, false))
            {
                assert!((y_c * (_x - center) - y).abs() < 1e-10);
            }
            g_c = g;
        }
    }
}
