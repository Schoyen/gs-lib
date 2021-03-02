use std::collections::HashMap;

use super::G1D;

pub struct OD1D<'a> {
    pub g_i: &'a G1D,
    pub g_j: &'a G1D,

    pub i: u32,
    pub j: u32,
    pub tot_exp: f64,
    red_exp: f64,

    center_diff: f64,
    pub com: f64,
    i_com: f64,
    j_com: f64,

    coefficients: HashMap<(i32, i32, i32), f64>,
}

impl<'a> OD1D<'a> {
    pub fn new(g_i: &'a G1D, g_j: &'a G1D) -> Self {
        let i = g_i.i;
        let j = g_j.i;
        let tot_exp = g_i.a + g_j.a; // p
        let red_exp = g_i.a * g_j.a / tot_exp; // mu
        let center_diff = g_i.center - g_j.center; // X_AB
        let com = (g_i.a * g_i.center + g_j.a * g_j.center) / tot_exp; // P
        let i_com = com - g_i.center; // X_PA
        let j_com = com - g_j.center; // X_PB
        let exp_weight = (-red_exp * center_diff.powi(2)).exp(); // K_AB
        let mut coefficients = HashMap::new();
        coefficients.insert((0, 0, 0), exp_weight);

        OD1D {
            g_i,
            g_j,
            i,
            j,
            tot_exp,
            red_exp,
            center_diff,
            com,
            i_com,
            j_com,
            coefficients,
        }
    }

    pub fn evaluate_point(&self, x: f64, with_norm: bool) -> f64 {
        self.g_i.evaluate_point(x, with_norm)
            * self.g_j.evaluate_point(x, with_norm)
    }

    pub fn evaluate(&self, x: &Vec<f64>, with_norm: bool) -> Vec<f64> {
        let mut res = vec![0.0; x.len()];

        for i in 0..x.len() {
            res[i] = self.evaluate_point(x[i], with_norm);
        }

        res
    }

    pub fn expansion_coefficients(&mut self, t: i32) -> f64 {
        self._expansion_coefficients(self.i as i32, self.j as i32, t)
    }

    pub fn _expansion_coefficients(&mut self, i: i32, j: i32, t: i32) -> f64 {
        let key = (i, j, t);

        if self.coefficients.contains_key(&key) {
            return *self.coefficients.get(&key).unwrap();
        }

        if t < 0 || t > (i + j) || i < 0 || j < 0 {
            return 0.0;
        }

        if i == 0 {
            let val = 1.0 / (2.0 * self.tot_exp)
                * self._expansion_coefficients(i, j - 1, t - 1)
                + self.j_com * self._expansion_coefficients(i, j - 1, t)
                + ((t + 1) as f64)
                    * self._expansion_coefficients(i, j - 1, t + 1);
            self.coefficients.insert((i, j, t), val);
        } else {
            let val = 1.0 / (2.0 * self.tot_exp)
                * self._expansion_coefficients(i - 1, j, t - 1)
                + self.i_com * self._expansion_coefficients(i - 1, j, t)
                + ((t + 1) as f64)
                    * self._expansion_coefficients(i - 1, j, t + 1);
            self.coefficients.insert((i, j, t), val);
        }

        return *self.coefficients.get(&key).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let g_0 = G1D::new(0, 1.0, 0.0, 'x');
        let g_1 = G1D::new(1, 1.0, 0.5, 'x');
        let od_01 = OD1D::new(&g_0, &g_1);

        assert_eq!(od_01.red_exp, g_0.a * g_1.a / (g_0.a + g_1.a));
        assert_eq!(od_01.center_diff, g_0.center - g_1.center);
    }

    #[test]
    fn test_expansion_coefficients() {
        let g_0 = G1D::new(0, 1.0, -0.5, 'x');
        let g_1 = G1D::new(1, 0.7, 0.5, 'x');
        let g_2 = G1D::new(2, 1.2, 0.3, 'x');
        let mut od_01 = OD1D::new(&g_0, &g_1);
        let mut od_02 = OD1D::new(&g_0, &g_2);
        let mut od_21 = OD1D::new(&g_2, &g_1);

        assert!((od_01.expansion_coefficients(0) - (-0.38969419)).abs() < 1e-7);
        assert!((od_01.expansion_coefficients(1) - 0.19484709).abs() < 1e-7);
        assert!((od_01.expansion_coefficients(2)).abs() < 1e-10);
        assert!((od_02.expansion_coefficients(0) - 0.25356869).abs() < 1e-7);
        assert!((od_02.expansion_coefficients(1) - (-0.11658330)).abs() < 1e-7);
        assert!((od_21.expansion_coefficients(0) - 0.00476926).abs() < 1e-7);
        assert!((od_21.expansion_coefficients(2) - 0.00143238).abs() < 1e-7);
    }
}
