use ndarray::{Array, Array4};
use rgsl::bessel::In_scaled;

use super::{G2D, OD2D};

pub fn construct_coulomb_operator_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array4<f64> {
    let l = gaussians.len();

    let mut u = Array::zeros((l, l, l, l));

    for a in 0..l {
        let g_a = &gaussians[a];

        for b in 0..l {
            let g_b = &gaussians[b];

            for c in 0..l {
                let g_c = &gaussians[c];

                for d in 0..l {
                    let g_d = &gaussians[d];

                    u[[a, b, c, d]] = g_a.norm
                        * g_b.norm
                        * g_c.norm
                        * g_d.norm
                        * construct_coulomb_operator_matrix_element(
                            g_a, g_b, g_c, g_d,
                        );
                }
            }
        }
    }

    u
}

pub fn construct_coulomb_operator_matrix_element(
    g_a: &G2D,
    g_b: &G2D,
    g_c: &G2D,
    g_d: &G2D,
) -> f64 {
    let mut od_ac = OD2D::new(g_a, g_c);
    let mut od_bd = OD2D::new(g_b, g_d);

    construct_coulomb_operator_matrix_element_od(&mut od_ac, &mut od_bd)
}

pub fn construct_coulomb_operator_matrix_element_od(
    od_ac: &mut OD2D,
    od_bd: &mut OD2D,
) -> f64 {
    let p = od_ac.tot_exp;
    let q = od_bd.tot_exp;

    let sigma = (p + q) / (4.0 * p * q);
    let delta = (
        od_bd.center_diff.0 - od_ac.center_diff.0,
        od_bd.center_diff.1 - od_ac.center_diff.1,
    );

    let mut val = 0.0;

    for t in 0..(od_ac.x_sum_lim + 1) {
        for u in 0..(od_ac.y_sum_lim + 1) {
            let e_ac = od_ac.expansion_coefficients(t as i32, u as i32);

            for tau in 0..(od_bd.x_sum_lim + 1) {
                for nu in 0..(od_bd.y_sum_lim + 1) {
                    let e_bd = (-1.0 as f64).powi((tau + nu) as i32)
                        * od_bd.expansion_coefficients(tau as i32, nu as i32);

                    val += e_ac
                        * e_bd
                        * int_tilde(
                            (t + tau) as i32,
                            (u + nu) as i32,
                            sigma,
                            delta,
                        );
                }
            }
        }
    }

    std::f64::consts::PI.powi(2) / (p * q)
        * (std::f64::consts::PI / (4.0 * sigma)).sqrt()
        * val
}

fn int_tilde(t: i32, u: i32, sigma: f64, delta: (f64, f64)) -> f64 {
    int_tilde_rec(0, t, u, sigma, delta)
}

fn int_tilde_rec(n: i32, t: i32, u: i32, sigma: f64, delta: (f64, f64)) -> f64 {
    assert!(n >= 0);

    if t < 0 || u < 0 {
        return 0.0;
    }

    if t == 0 && u == 0 {
        return extended_bessel(n, sigma, delta);
    }

    let mut pre_factor = 1.0 / (8.0 * sigma);
    let mut val = 0.0;

    if n == 0 {
        pre_factor *= 2.0;

        if t == 0 {
            val += delta.1
                * (int_tilde_rec(n, t, u - 1, sigma, delta)
                    + int_tilde_rec(n + 1, t, u - 1, sigma, delta));

            if u > 1 {
                val += -((u - 1) as f64)
                    * (int_tilde_rec(n, t, u - 2, sigma, delta)
                        + int_tilde_rec(n + 1, t, u - 2, sigma, delta));
            }

            return val * pre_factor;
        }

        val += delta.0
            * (int_tilde_rec(n, t - 1, u, sigma, delta)
                + int_tilde_rec(n + 1, t - 1, u, sigma, delta));

        if t > 1 {
            val += -((t - 1) as f64)
                * (int_tilde_rec(n, t - 2, u, sigma, delta)
                    + int_tilde_rec(n + 1, t - 2, u, sigma, delta));
        }

        return val * pre_factor;
    }

    if t == 0 {
        val += delta.1
            * (int_tilde_rec(n - 1, t, u - 1, sigma, delta)
                + 2.0 * int_tilde_rec(n, t, u - 1, sigma, delta)
                + int_tilde_rec(n + 1, t, u - 1, sigma, delta));

        if u > 1 {
            val += -((u - 1) as f64)
                * (int_tilde_rec(n - 1, t, u - 2, sigma, delta)
                    + 2.0 * int_tilde_rec(n, t, u - 2, sigma, delta)
                    + int_tilde_rec(n + 1, t, u - 2, sigma, delta));
        }

        return val * pre_factor;
    }

    val += delta.0
        * (int_tilde_rec(n - 1, t - 1, u, sigma, delta)
            + 2.0 * int_tilde_rec(n, t - 1, u, sigma, delta)
            + int_tilde_rec(n + 1, t - 1, u, sigma, delta));

    if t > 1 {
        val += -((t - 1) as f64)
            * (int_tilde_rec(n - 1, t - 2, u, sigma, delta)
                + 2.0 * int_tilde_rec(n, t - 2, u, sigma, delta)
                + int_tilde_rec(n + 1, t - 2, u, sigma, delta));
    }

    val * pre_factor
}

fn extended_bessel(n: i32, sigma: f64, delta: (f64, f64)) -> f64 {
    let delta_sq = delta.0.powi(2) + delta.1.powi(2);
    let arg = -delta_sq / (8.0 * sigma);

    In_scaled(n, arg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_extended_bessel() {
        let delta = (2.3, 0.7);
        let sigma = 0.6;

        assert_abs_diff_eq!(
            extended_bessel(0, sigma, delta),
            0.4189318801994202,
        );
        assert_abs_diff_eq!(
            extended_bessel(1, sigma, delta),
            -0.21536075261087415,
        );
        assert_abs_diff_eq!(
            extended_bessel(2, sigma, delta),
            0.06123928070731078,
        );
        assert_abs_diff_eq!(
            extended_bessel(3, sigma, delta),
            -0.011936152337454205,
        );
        assert_abs_diff_eq!(
            extended_bessel(4, sigma, delta),
            0.0017650268459473005,
        );
    }

    #[test]
    fn test_transpose_u() {
        let gaussians = vec![
            G2D::new((0, 0), 1.0, (0.0, 0.0)),
            G2D::new((1, 0), 1.0, (0.0, 0.0)),
            G2D::new((0, 1), 1.0, (0.0, 0.0)),
            G2D::new((1, 1), 1.0, (0.0, 0.0)),
        ];

        let u = construct_coulomb_operator_matrix_elements(&gaussians);
        let l = gaussians.len();

        assert!(u.shape().len() == 4);

        for &dim_len in u.shape().into_iter() {
            assert!(dim_len == l);
        }

        for p in 0..l {
            for q in 0..l {
                for r in 0..l {
                    for s in 0..l {
                        assert_abs_diff_eq!(u[[p, q, r, s]], u[[r, s, p, q]]);
                    }
                }
            }
        }
    }
}
