use super::{G1D, OD1D};
use super::s;

use ndarray::{Array, Array2};


pub fn construct_kinetic_matrix_elements(
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    -0.5 * construct_differential_operator_matrix_elements(2, gaussians)
}


pub fn construct_differential_operator_matrix_elements(
    e: u32, gaussians: &Vec<G1D>,
) -> Array2<f64> {
    let l = gaussians.len();
    let mut d_e = Array::zeros((l, l));

    for i in 0..l {
        let g_i = &gaussians[i];
        d_e[[i, i]] = g_i.norm.powi(2) * d(e, g_i, g_i);

        for j in (i + 1)..l {
            let g_j = &gaussians[j];
            let val = g_i.norm * g_j.norm * d(e, g_i, g_j);

            d_e[[i, j]] = val;
            d_e[[j, i]] = val;
        }
    }

    d_e
}


fn d(e: u32, g_i: &G1D, g_j: &G1D) -> f64 {
    if e == 0 {
        return s(0, 0.0, OD1D::new(g_i, g_j));
    }

    let i = g_i.i;
    let a = g_i.a;
    let center = g_i.center;

    let forward = 2.0 * a * d(e - 1, &G1D::new(i + 1, a, center, g_i.symbol), g_j);
    let backward = if i < 1 { 0.0 } else { -(i as f64) * d(e - 1, &G1D::new(i - 1, a, center, g_i.symbol), g_j) };

    forward + backward
}
