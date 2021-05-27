use super::{G1D, OD1D};

use ndarray::{Array, Array2};

pub fn construct_gaussian_operator_matrix_elements(
    op: &G1D,
    gaussians: &Vec<G1D>,
) -> Array2<f64> {
    let l = gaussians.len();
    let mut gop_k = Array::zeros((l, l));

    for i in 0..l {
        let g_i = &gaussians[i];
        gop_k[[i, i]] = g_i.norm.powi(2) * g_op(op, g_i, g_i);

        for j in (i + 1)..l {
            let g_j = &gaussians[j];
            let val = g_i.norm * g_j.norm * g_op(op, g_i, g_j);

            gop_k[[i, j]] = val;
            gop_k[[j, i]] = val;
        }
    }

    gop_k
}

fn g_op(op: &G1D, g_i: &G1D, g_j: &G1D) -> f64 {
    let mut val = 0.0;
    let mut od = OD1D::new(g_i, g_j);
    let g_0 = G1D::new(0, od.tot_exp, od.com, g_i.symbol);

    for t in 0..(od.i + od.j + 1) {
        val += od.expansion_coefficients(t as i32) * p_rec(t, op, &g_0);
    }

    val
}

fn p_rec(t: u32, op: &G1D, g_l: &G1D) -> f64 {
    if t == 0 {
        let mut od = OD1D::new(op, g_l);
        return od.expansion_coefficients(0)
            * (std::f64::consts::PI / od.tot_exp).sqrt();
    }

    2.0 * g_l.a * p_rec(t - 1, op, &g_l.increment_i())
        - (if g_l.i == 0 {
            0.0
        } else {
            (g_l.i as f64) * p_rec(t - 1, op, &g_l.decrement_i())
        })
}
