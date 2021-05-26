use super::{G1D, OD1D};

use ndarray::{Array, Array2};

pub fn construct_overlap_matrix_elements(gaussians: &Vec<G1D>) -> Array2<f64> {
    construct_diff_mm_matrix_elements(0, 0, 0.0, gaussians)
}

pub fn construct_differential_operator_matrix_elements(
    f: u32,
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    construct_diff_mm_matrix_elements(0, f, 0.0, gaussians)
}

pub fn construct_kinetic_operator_matrix_elements(
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    -0.5 * construct_diff_mm_matrix_elements(0, 2, 0.0, gaussians)
}

pub fn construct_multipole_moment_matrix_elements(
    e: u32,
    center: f64,
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    construct_diff_mm_matrix_elements(e, 0, center, gaussians)
}

pub fn construct_diff_mm_matrix_elements(
    e: u32,
    f: u32,
    center: f64,
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    let l = gaussians.len();
    let mut l_ef = Array::zeros((l, l));

    for i in 0..l {
        let g_i = &gaussians[i];
        l_ef[[i, i]] = g_i.norm.powi(2) * l_rec(e, f, center, g_i, g_i);

        for j in (i + 1)..l {
            let g_j = &gaussians[j];
            let val = g_i.norm * g_j.norm * l_rec(e, f, center, g_i, g_j);

            l_ef[[i, j]] = val;
            l_ef[[j, i]] = val;
        }
    }

    l_ef
}

pub fn l_rec(e: u32, f: u32, center: f64, g_i: &G1D, g_j: &G1D) -> f64 {
    if e == 0 && f == 0 {
        let mut od = OD1D::new(g_i, g_j);
        return od.expansion_coefficients(0)
            * (std::f64::consts::PI / od.tot_exp).sqrt();
    }

    if e == 0 {
        return (g_j.i as f64)
            * (if g_j.i > 0 {
                l_rec(e, f - 1, center, g_i, &g_j.decrement_i())
            } else {
                0.0
            } - 2.0
                * g_j.a
                * l_rec(e, f - 1, center, g_i, &g_j.increment_i()));
    }

    l_rec(e - 1, f, center, &g_i.increment_i(), g_j)
        + (g_i.center - center) * l_rec(e - 1, f, center, g_i, g_j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    fn trapz(f: &Array1<f64>, x: &Array1<f64>) -> f64 {
        let dx = x[1] - x[0];
        let mut val = 0.0;

        for i in 1..x.len() {
            val += f[i - 1] + f[i]
        }

        0.5 * val * dx
    }

    fn construct_overlap_matrix_elements(gaussians: &Vec<G1D>) -> Array2<f64> {
        tests::construct_multipole_moment_matrix_elements(0, 0.0, gaussians)
    }

    fn construct_multipole_moment_matrix_elements(
        e: u32,
        center: f64,
        gaussians: &Vec<G1D>,
    ) -> Array2<f64> {
        let l = gaussians.len();
        let mut s_e = Array::zeros((l, l));

        for i in 0..l {
            let g_i = &gaussians[i];
            s_e[[i, i]] = g_i.norm.powi(2)
                * tests::s(e, center, &mut OD1D::new(&g_i, &g_i));

            for j in (i + 1)..l {
                let g_j = &gaussians[j];

                let val = g_i.norm
                    * g_j.norm
                    * tests::s(e, center, &mut OD1D::new(&g_i, &g_j));

                s_e[[i, j]] = val;
                s_e[[j, i]] = val;
            }
        }

        s_e
    }

    fn s(e: u32, center: f64, od: &mut OD1D) -> f64 {
        let mut val = 0.0;

        for t in 0..(std::cmp::min(od.i + od.j, e) + 1) {
            val += od.expansion_coefficients(t as i32)
                * tests::m(e as i32, t as i32, od.tot_exp, od.com, center);
        }

        val
    }

    fn m(e: i32, t: i32, p: f64, od_center: f64, center: f64) -> f64 {
        if t > e {
            return 0.0;
        }

        if t < 0 || e < 0 {
            return 0.0;
        }

        if e == 0 {
            return if t == 0 {
                (std::f64::consts::PI / p).sqrt()
            } else {
                0.0
            };
        }

        (t as f64) * tests::m(e - 1, t - 1, p, od_center, center)
            + (od_center - center) * tests::m(e - 1, t, p, od_center, center)
            + 1.0 / (2.0 * p) * tests::m(e - 1, t + 1, p, od_center, center)
    }

    #[test]
    fn test_trapz() {
        let x = Array::linspace(-10.0, 10.0, 1001);
        let x_num_sq = tests::trapz(&x, &x);

        assert_abs_diff_eq!(x_num_sq, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_overlap() {
        let g_list =
            vec![G1D::new(0, 2.0, 2.0, 'x'), G1D::new(0, 2.0, -2.0, 'x')];

        let s = tests::construct_overlap_matrix_elements(&g_list);
        let s_2 = construct_overlap_matrix_elements(&g_list);

        assert_eq!(s.nrows(), s_2.nrows());
        assert_eq!(s.ncols(), s_2.ncols());

        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert_abs_diff_eq!(s[[i, j]], s_2[[i, j]], epsilon = 1e-12);
            }
        }

        let gaussians = vec![
            G1D::new(0, 1.0, 0.5, 'x'),
            G1D::new(0, 0.5, 0.0, 'x'),
            G1D::new(1, 1.0, 0.0, 'x'),
            G1D::new(2, 1.0, 0.0, 'x'),
        ];

        let s = construct_overlap_matrix_elements(&gaussians);
        let s_2 = tests::construct_overlap_matrix_elements(&gaussians);

        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert_abs_diff_eq!(s[[i, j]], s_2[[i, j]], epsilon = 1e-12);
            }
        }

        let mut s_num = Array2::zeros((gaussians.len(), gaussians.len()));
        let x = Array::linspace(-10.0, 10.0, 1001);

        for i in 0..gaussians.len() {
            let g_i = &gaussians[i];
            let g_i_eval = g_i.evaluate(&x, false);

            for j in 0..gaussians.len() {
                let g_j = &gaussians[j];

                s_num[[i, j]] = g_i.norm
                    * g_j.norm
                    * tests::trapz(&(&g_i_eval * &g_j.evaluate(&x, false)), &x);
            }
        }

        for (s_o, s_n) in s.iter().zip(s_num.iter()) {
            assert_abs_diff_eq!(s_o, s_n, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_multipole_moment_matrix_elements() {
        let g_list = vec![
            G1D::new(0, 2.0, -4.0, 'x'),
            G1D::new(0, 2.0, 4.0, 'x'),
            G1D::new(1, 1.0, 0.0, 'x'),
            G1D::new(2, 1.0, 0.0, 'x'),
        ];

        let d =
            tests::construct_multipole_moment_matrix_elements(1, 1.0, &g_list);
        let d_2 = construct_multipole_moment_matrix_elements(1, 1.0, &g_list);

        assert_eq!(d.nrows(), d_2.nrows());
        assert_eq!(d.ncols(), d_2.ncols());

        for i in 0..d.nrows() {
            for j in 0..d.ncols() {
                assert_abs_diff_eq!(d[[i, j]], d[[j, i]], epsilon = 1e-12);
                assert_abs_diff_eq!(d[[i, j]], d_2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    fn construct_kinetic_operator_matrix_elements(
        gaussians: &Vec<G1D>,
    ) -> Array2<f64> {
        -0.5 * tests::construct_differential_operator_matrix_elements(
            2, gaussians,
        )
    }

    fn construct_differential_operator_matrix_elements(
        e: u32,
        gaussians: &Vec<G1D>,
    ) -> Array2<f64> {
        let l = gaussians.len();
        let mut d_e = Array::zeros((l, l));

        for i in 0..l {
            let g_i = &gaussians[i];
            d_e[[i, i]] = g_i.norm.powi(2) * tests::d(e, g_i, g_i);

            for j in (i + 1)..l {
                let g_j = &gaussians[j];
                let val = g_i.norm * g_j.norm * tests::d(e, g_i, g_j);

                d_e[[i, j]] = val;
                d_e[[j, i]] = val;
            }
        }

        d_e
    }

    fn d(e: u32, g_i: &G1D, g_j: &G1D) -> f64 {
        if e == 0 {
            return tests::s(0, 0.0, &mut OD1D::new(g_i, g_j));
        }

        let i = g_i.i;
        let a = g_i.a;

        let forward = 2.0 * a * tests::d(e - 1, &g_i.increment_i(), g_j);
        let backward = if i < 1 {
            0.0
        } else {
            -(i as f64) * tests::d(e - 1, &g_i.decrement_i(), g_j)
        };

        forward + backward
    }

    #[test]
    fn test_kinetic_energy_operator() {
        let gaussians = vec![
            G1D::new(0, 1.0, 0.5, 'x'),
            G1D::new(0, 0.5, 0.0, 'x'),
            G1D::new(1, 1.0, 0.0, 'x'),
            G1D::new(2, 1.0, 0.0, 'x'),
        ];

        let t = tests::construct_kinetic_operator_matrix_elements(&gaussians);
        assert!(t.is_square());

        for i in 0..t.nrows() {
            for j in 0..t.ncols() {
                assert_abs_diff_eq!(t[[i, j]], t[[j, i]]);
            }
        }
    }
}
