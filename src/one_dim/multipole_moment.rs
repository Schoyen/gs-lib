use super::{G1D, OD1D};

use ndarray::{Array, Array1, Array2};

pub fn trapz(f: &Array1<f64>, x: &Array1<f64>) -> f64 {
    let dx = x[1] - x[0];
    let mut val = 0.0;

    for i in 1..x.len() {
        val += f[i - 1] + f[i]
    }

    0.5 * val * dx
}

pub fn construct_overlap_matrix_elements(gaussians: &Vec<G1D>) -> Array2<f64> {
    construct_multipole_moment_matrix_elements(0, 0.0, gaussians)
}

fn construct_overlap_matrix_elements_inplace(
    gaussians: &Vec<G1D>,
    s: &mut Array2<f64>,
) {
    construct_multipole_moment_matrix_elements_inplace(0, 0.0, gaussians, s);
}

pub fn construct_multipole_moment_matrix_elements(
    e: u32,
    center: f64,
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    let l = gaussians.len();
    let mut s_e = Array::zeros((l, l));

    construct_multipole_moment_matrix_elements_inplace(
        e, center, gaussians, &mut s_e,
    );

    s_e
}

fn construct_multipole_moment_matrix_elements_inplace(
    e: u32,
    center: f64,
    gaussians: &Vec<G1D>,
    s_e: &mut Array2<f64>,
) {
    let l = gaussians.len();

    assert!(s_e.nrows() == l);
    assert!(s_e.is_square());

    for i in 0..l {
        let g_i = &gaussians[i];

        s_e[[i, i]] = g_i.norm.powi(2) * s(e, center, OD1D::new(&g_i, &g_i));

        for j in (i + 1)..l {
            let g_j = &gaussians[j];

            let val = g_i.norm * g_j.norm * s(e, center, OD1D::new(&g_i, &g_j));

            s_e[[i, j]] = val;
            s_e[[j, i]] = val;
        }
    }
}

pub fn s(e: u32, center: f64, od: OD1D) -> f64 {
    let mut val = 0.0;

    for t in 0..(std::cmp::min(od.i + od.j, e) + 1) {
        val += od.expansion_coefficients(t as i32)
            * m(e as i32, t as i32, od.tot_exp, od.com, center);
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

    (t as f64) * m(e - 1, t - 1, p, od_center, center)
        + (od_center - center) * m(e - 1, t, p, od_center, center)
        + 1.0 / (2.0 * p) * m(e - 1, t + 1, p, od_center, center)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_trapz() {
        let x = Array::linspace(-10.0, 10.0, 1001);
        let x_num_sq = trapz(&x, &x);

        assert_abs_diff_eq!(x_num_sq, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_overlap() {
        let g_list =
            vec![G1D::new(0, 2.0, 2.0, 'x'), G1D::new(0, 2.0, -2.0, 'x')];

        let s = construct_overlap_matrix_elements(&g_list);
        let mut s_2 = Array2::zeros((g_list.len(), g_list.len()));
        construct_overlap_matrix_elements_inplace(&g_list, &mut s_2);

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
        let mut s_num = Array2::zeros((gaussians.len(), gaussians.len()));
        let x = Array::linspace(-10.0, 10.0, 1001);

        for i in 0..gaussians.len() {
            let g_i = &gaussians[i];
            let g_i_eval = g_i.evaluate(&x, false);

            for j in 0..gaussians.len() {
                let g_j = &gaussians[j];

                s_num[[i, j]] = g_i.norm
                    * g_j.norm
                    * trapz(&(&g_i_eval * &g_j.evaluate(&x, false)), &x);
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

        let d = construct_multipole_moment_matrix_elements(1, 1.0, &g_list);
        let mut d_2 = Array2::zeros((g_list.len(), g_list.len()));
        construct_multipole_moment_matrix_elements_inplace(
            1, 1.0, &g_list, &mut d_2,
        );

        assert_eq!(d.nrows(), d_2.nrows());
        assert_eq!(d.ncols(), d_2.ncols());

        for i in 0..d.nrows() {
            for j in 0..d.ncols() {
                assert_abs_diff_eq!(d[[i, j]], d[[j, i]], epsilon = 1e-12);
                assert_abs_diff_eq!(d[[i, j]], d_2[[i, j]], epsilon = 1e-12);
            }
        }
    }
}
