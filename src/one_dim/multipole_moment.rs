use super::G1D;
use super::OD1D;

use ndarray::{Array, Array2};

fn construct_overlap_matrix_elements(gaussians: &Vec<G1D>) -> Array2<f64> {
    construct_multipole_moment_matrix_elements(0, 0.0, gaussians)
}

fn construct_overlap_matrix_elements_inplace(
    gaussians: &Vec<G1D>,
    s: &mut Array2<f64>,
) {
    construct_multipole_moment_matrix_elements_inplace(0, 0.0, gaussians, s);
}

fn construct_multipole_moment_matrix_elements(
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

fn s(e: u32, center: f64, od: OD1D) -> f64 {
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
            0.0
        } else {
            (std::f64::consts::PI / p).sqrt()
        };
    }

    (t as f64) * m(e - 1, t - 1, p, od_center, center)
        + (od_center - center) * m(e - 1, t, p, od_center, center)
        + 1.0 / (2.0 * p) * m(e - 1, t + 1, p, od_center, center)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap() {
        let g_list =
            vec![G1D::new(0, 2.0, 2.0, 'x'), G1D::new(0, 2.0, -2.0, 'x')];

        let s = construct_overlap_matrix_elements(&g_list);

        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert!(s[[i, j]].abs() < 1e-12);
            }
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

        for i in 0..d.nrows() {
            for j in 0..d.ncols() {
                assert!((d[[i, j]] - d[[j, i]]).abs() < 1e-12);
            }
        }
    }
}
