use ndarray::{Array, Array4};
use rgsl::bessel::In_scaled;

use super::{G2D, OD2D};

pub fn construct_coulomb_operator_matrix_elements(
    gaussians: &Vec<G2D>,
) -> Array4<f64> {
    let l = gaussians.len();

    let mut u = Array::zeros((l, l, l, l));
    // let mut counter = 0u64;

    for a in 0..l {
        let g_a = &gaussians[a];
        // TODO: Is it possible to reuse od_aa instead of making a copy? This
        // would require some handling of two mutable references.
        let mut od_aa = OD2D::new(g_a, g_a);
        let mut od_aa_c = OD2D::new(g_a, g_a);

        u[[a, a, a, a]] = g_a.norm.powi(4)
            * construct_coulomb_operator_matrix_element_od(
                &mut od_aa,
                &mut od_aa_c,
            );
        // counter += 1;

        for b in 0..l {
            if b == a {
                continue;
            }

            let g_b = &gaussians[b];
            let mut od_ab = OD2D::new(g_a, g_b);
            let mut od_ba = OD2D::new(g_b, g_a);
            let mut od_bb = OD2D::new(g_b, g_b);

            let val = g_a.norm
                * g_b.norm.powi(3)
                * construct_coulomb_operator_matrix_element_od(
                    &mut od_ab, &mut od_bb,
                );
            // counter += 1;

            u[[a, b, b, b]] = val;
            u[[b, a, b, b]] = val;
            u[[b, b, a, b]] = val;
            u[[b, b, b, a]] = val;

            let val = g_a.norm.powi(2)
                * g_b.norm.powi(2)
                * construct_coulomb_operator_matrix_element_od(
                    &mut od_aa, &mut od_bb,
                );

            u[[a, b, a, b]] = val;
            u[[b, a, b, a]] = val;

            let val = g_a.norm.powi(2)
                * g_b.norm.powi(2)
                * construct_coulomb_operator_matrix_element_od(
                    &mut od_ab, &mut od_ba,
                );
            // counter += 1;

            u[[a, a, b, b]] = val;
            u[[b, b, a, a]] = val;
            u[[a, b, b, a]] = val;
            u[[b, a, a, b]] = val;

            for c in 0..l {
                if c == b || c == a {
                    continue;
                }

                let g_c = &gaussians[c];

                let mut od_bc = OD2D::new(g_b, g_c);
                let mut od_ac = OD2D::new(g_a, g_c);

                let val = g_a.norm.powi(2)
                    * g_b.norm
                    * g_c.norm
                    * construct_coulomb_operator_matrix_element_od(
                        &mut od_aa, &mut od_bc,
                    );
                // counter += 1;

                u[[a, b, a, c]] = val;
                u[[b, a, c, a]] = val;
                u[[a, c, a, b]] = val;
                u[[c, a, b, a]] = val;

                let val = g_a.norm.powi(2)
                    * g_b.norm
                    * g_c.norm
                    * construct_coulomb_operator_matrix_element_od(
                        &mut od_ab, &mut od_ac,
                    );
                // counter += 1;

                u[[a, a, b, c]] = val;
                u[[a, a, c, b]] = val;
                u[[b, c, a, a]] = val;
                u[[c, b, a, a]] = val;
                u[[a, b, c, a]] = val;
                u[[b, a, a, c]] = val;
                u[[c, a, a, b]] = val;
                u[[a, c, b, a]] = val;

                for d in 0..l {
                    if d == c || d == b || d == a {
                        continue;
                    }

                    let g_d = &gaussians[d];
                    let mut od_bd = OD2D::new(g_b, g_d);

                    let val = g_a.norm
                        * g_b.norm
                        * g_c.norm
                        * g_d.norm
                        * construct_coulomb_operator_matrix_element_od(
                            &mut od_ac, &mut od_bd,
                        );
                    // counter += 1;

                    u[[a, b, c, d]] = val;
                    u[[b, a, d, c]] = val;
                    u[[c, d, a, b]] = val;
                    u[[d, c, b, a]] = val;
                }
            }
        }
    }
    // dbg!(counter);

    u
}

pub fn construct_coulomb_operator_matrix_element_od(
    od_ac: &mut OD2D,
    od_bd: &mut OD2D,
) -> f64 {
    let p = od_ac.tot_exp;
    let q = od_bd.tot_exp;

    let sigma = (p + q) / (4.0 * p * q);
    let delta = (od_bd.com.0 - od_ac.com.0, od_bd.com.1 - od_ac.com.1);

    let mut val = 0.0;

    for t in 0..(od_ac.x_sum_lim + 1) {
        for u in 0..(od_ac.y_sum_lim + 1) {
            let e_ac = (-1.0f64).powi((t + u) as i32)
                * od_ac.expansion_coefficients(t as i32, u as i32);

            // Note: Hacky optimization. This needs to be tested
            if e_ac.abs() < 1e-12 {
                continue;
            }

            for tau in 0..(od_bd.x_sum_lim + 1) {
                for nu in 0..(od_bd.y_sum_lim + 1) {
                    let e_bd =
                        od_bd.expansion_coefficients(tau as i32, nu as i32);

                    // Note: Hacky optimization. This needs to be tested
                    if e_bd.abs() < 1e-12 {
                        continue;
                    }

                    val += e_ac
                        * e_bd
                        * int_tilde_q(
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

fn int_tilde_q(t: i32, u: i32, sigma: f64, delta: (f64, f64)) -> f64 {
    int_tilde_rec_q(0, t, u, sigma, delta)
}

fn int_tilde_rec_q(
    n: i32,
    t: i32,
    u: i32,
    sigma: f64,
    delta: (f64, f64),
) -> f64 {
    assert!(n >= 0);

    if t < 0 || u < 0 {
        return 0.0;
    }

    if t == 0 && u == 0 {
        return extended_bessel(n, sigma, delta);
    }

    let mut pre_factor = -1.0 / (8.0 * sigma);
    let mut val = 0.0;

    if n == 0 {
        pre_factor *= 2.0;

        if t == 0 {
            val += delta.1
                * (int_tilde_rec_q(n, t, u - 1, sigma, delta)
                    + int_tilde_rec_q(n + 1, t, u - 1, sigma, delta));

            if u > 1 {
                val += ((u - 1) as f64)
                    * (int_tilde_rec_q(n, t, u - 2, sigma, delta)
                        + int_tilde_rec_q(n + 1, t, u - 2, sigma, delta));
            }

            return val * pre_factor;
        }

        val += delta.0
            * (int_tilde_rec_q(n, t - 1, u, sigma, delta)
                + int_tilde_rec_q(n + 1, t - 1, u, sigma, delta));

        if t > 1 {
            val += ((t - 1) as f64)
                * (int_tilde_rec_q(n, t - 2, u, sigma, delta)
                    + int_tilde_rec_q(n + 1, t - 2, u, sigma, delta));
        }

        return val * pre_factor;
    }

    if t == 0 {
        val += delta.1
            * (int_tilde_rec_q(n - 1, t, u - 1, sigma, delta)
                + 2.0 * int_tilde_rec_q(n, t, u - 1, sigma, delta)
                + int_tilde_rec_q(n + 1, t, u - 1, sigma, delta));

        if u > 1 {
            val += ((u - 1) as f64)
                * (int_tilde_rec_q(n - 1, t, u - 2, sigma, delta)
                    + 2.0 * int_tilde_rec_q(n, t, u - 2, sigma, delta)
                    + int_tilde_rec_q(n + 1, t, u - 2, sigma, delta));
        }

        return val * pre_factor;
    }

    val += delta.0
        * (int_tilde_rec_q(n - 1, t - 1, u, sigma, delta)
            + 2.0 * int_tilde_rec_q(n, t - 1, u, sigma, delta)
            + int_tilde_rec_q(n + 1, t - 1, u, sigma, delta));

    if t > 1 {
        val += ((t - 1) as f64)
            * (int_tilde_rec_q(n - 1, t - 2, u, sigma, delta)
                + 2.0 * int_tilde_rec_q(n, t - 2, u, sigma, delta)
                + int_tilde_rec_q(n + 1, t - 2, u, sigma, delta));
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
    use rand::Rng;

    fn construct_coulomb_operator_matrix_element(
        g_a: &G2D,
        g_b: &G2D,
        g_c: &G2D,
        g_d: &G2D,
    ) -> f64 {
        let mut od_ac = OD2D::new(g_a, g_c);
        let mut od_bd = OD2D::new(g_b, g_d);

        construct_coulomb_operator_matrix_element_od(&mut od_ac, &mut od_bd)
    }

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
        assert_abs_diff_eq!(
            extended_bessel(3, sigma, delta),
            extended_bessel(3, sigma, (-delta.0, -delta.1)),
        );
        assert_abs_diff_eq!(
            extended_bessel(4, sigma, delta),
            extended_bessel(4, sigma, (-delta.0, -delta.1)),
        );
    }

    fn int_tilde(t: i32, u: i32, sigma: f64, delta: (f64, f64)) -> f64 {
        tests::int_tilde_rec(0, t, u, sigma, delta)
    }

    fn int_tilde_rec(
        n: i32,
        t: i32,
        u: i32,
        sigma: f64,
        delta: (f64, f64),
    ) -> f64 {
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
                    * (tests::int_tilde_rec(n, t, u - 1, sigma, delta)
                        + tests::int_tilde_rec(n + 1, t, u - 1, sigma, delta));

                if u > 1 {
                    val += -((u - 1) as f64)
                        * (tests::int_tilde_rec(n, t, u - 2, sigma, delta)
                            + tests::int_tilde_rec(
                                n + 1,
                                t,
                                u - 2,
                                sigma,
                                delta,
                            ));
                }

                return val * pre_factor;
            }

            val += delta.0
                * (tests::int_tilde_rec(n, t - 1, u, sigma, delta)
                    + tests::int_tilde_rec(n + 1, t - 1, u, sigma, delta));

            if t > 1 {
                val += -((t - 1) as f64)
                    * (tests::int_tilde_rec(n, t - 2, u, sigma, delta)
                        + tests::int_tilde_rec(n + 1, t - 2, u, sigma, delta));
            }

            return val * pre_factor;
        }

        if t == 0 {
            val += delta.1
                * (tests::int_tilde_rec(n - 1, t, u - 1, sigma, delta)
                    + 2.0 * tests::int_tilde_rec(n, t, u - 1, sigma, delta)
                    + tests::int_tilde_rec(n + 1, t, u - 1, sigma, delta));

            if u > 1 {
                val += -((u - 1) as f64)
                    * (tests::int_tilde_rec(n - 1, t, u - 2, sigma, delta)
                        + 2.0
                            * tests::int_tilde_rec(n, t, u - 2, sigma, delta)
                        + tests::int_tilde_rec(n + 1, t, u - 2, sigma, delta));
            }

            return val * pre_factor;
        }

        val += delta.0
            * (tests::int_tilde_rec(n - 1, t - 1, u, sigma, delta)
                + 2.0 * tests::int_tilde_rec(n, t - 1, u, sigma, delta)
                + tests::int_tilde_rec(n + 1, t - 1, u, sigma, delta));

        if t > 1 {
            val += -((t - 1) as f64)
                * (tests::int_tilde_rec(n - 1, t - 2, u, sigma, delta)
                    + 2.0 * tests::int_tilde_rec(n, t - 2, u, sigma, delta)
                    + tests::int_tilde_rec(n + 1, t - 2, u, sigma, delta));
        }

        val * pre_factor
    }

    #[test]
    fn compare_int_tilde_pq() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..4 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..2), rng.gen_range(0..2)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        let mut od_ac = OD2D::new(&gaussians[0], &gaussians[2]);
        let mut od_bd = OD2D::new(&gaussians[1], &gaussians[3]);

        let p_r = od_ac.tot_exp;
        let q_r = od_bd.tot_exp;

        let sigma_r = (p_r + q_r) / (4.0 * p_r * q_r);
        let delta_r = (od_bd.com.0 - od_ac.com.0, od_bd.com.1 - od_ac.com.1);

        for t in 0..(od_ac.x_sum_lim + 1) {
            for u in 0..(od_ac.y_sum_lim + 1) {
                let e_ac_q = (-1.0f64).powi((t + u) as i32)
                    * od_ac.expansion_coefficients(t as i32, u as i32);
                let e_ac = od_ac.expansion_coefficients(t as i32, u as i32);

                for tau in 0..(od_bd.x_sum_lim + 1) {
                    for nu in 0..(od_bd.y_sum_lim + 1) {
                        let e_bd_q =
                            od_bd.expansion_coefficients(tau as i32, nu as i32);
                        let e_bd = (-1.0f64).powi((tau + nu) as i32)
                            * od_bd
                                .expansion_coefficients(tau as i32, nu as i32);

                        let val_q = e_ac_q
                            * e_bd_q
                            * int_tilde_q(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_r,
                                delta_r,
                            );
                        let val = e_ac
                            * e_bd
                            * tests::int_tilde(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_r,
                                delta_r,
                            );
                        assert_abs_diff_eq!(val_q, val);
                    }
                }
            }
        }
    }

    #[test]
    fn test_int_tilde() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..4 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..2), rng.gen_range(0..2)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        let mut od_ac = OD2D::new(&gaussians[0], &gaussians[2]);
        let mut od_bd = OD2D::new(&gaussians[1], &gaussians[3]);
        let mut od_ca = OD2D::new(&gaussians[2], &gaussians[0]);
        let mut od_db = OD2D::new(&gaussians[3], &gaussians[1]);

        let p_r = od_ac.tot_exp;
        let q_r = od_bd.tot_exp;
        let p_l = od_ca.tot_exp;
        let q_l = od_db.tot_exp;

        let sigma_r = (p_r + q_r) / (4.0 * p_r * q_r);
        let sigma_l = (p_l + q_l) / (4.0 * p_l * q_l);

        let delta_rr = (od_bd.com.0 - od_ac.com.0, od_bd.com.1 - od_ac.com.1);
        let delta_lr = (od_bd.com.0 - od_ca.com.0, od_bd.com.1 - od_ca.com.1);
        let delta_rl = (od_db.com.0 - od_ac.com.0, od_db.com.1 - od_ac.com.1);
        let delta_ll = (od_db.com.0 - od_ca.com.0, od_db.com.1 - od_ca.com.1);

        assert_abs_diff_eq!(p_r, p_l);
        assert_abs_diff_eq!(q_r, q_l);
        assert_abs_diff_eq!(sigma_r, sigma_l);
        assert_abs_diff_eq!(delta_rr.0, delta_ll.0);
        assert_abs_diff_eq!(delta_rr.1, delta_ll.1);
        assert_abs_diff_eq!(delta_rr.0, delta_lr.0);
        assert_abs_diff_eq!(delta_rr.1, delta_lr.1);
        assert_abs_diff_eq!(delta_rr.0, delta_rl.0);
        assert_abs_diff_eq!(delta_rr.1, delta_rl.1);

        assert!(od_ac.x_sum_lim == od_ca.x_sum_lim);
        assert!(od_ac.y_sum_lim == od_ca.y_sum_lim);
        assert!(od_bd.x_sum_lim == od_db.x_sum_lim);
        assert!(od_bd.y_sum_lim == od_db.y_sum_lim);

        let mut val_rr = 0.0;
        let mut val_lr = 0.0;
        let mut val_rl = 0.0;
        let mut val_ll = 0.0;

        for t in 0..(od_ac.x_sum_lim + 1) {
            for u in 0..(od_ac.y_sum_lim + 1) {
                let tu_sign = (-1.0f64).powi((t + u) as i32);
                let e_ac =
                    tu_sign * od_ac.expansion_coefficients(t as i32, u as i32);
                let e_ca =
                    tu_sign * od_ca.expansion_coefficients(t as i32, u as i32);

                assert_abs_diff_eq!(e_ac, e_ca);

                for tau in 0..(od_bd.x_sum_lim + 1) {
                    for nu in 0..(od_bd.y_sum_lim + 1) {
                        let e_bd =
                            od_bd.expansion_coefficients(tau as i32, nu as i32);
                        let e_db =
                            od_db.expansion_coefficients(tau as i32, nu as i32);

                        assert_abs_diff_eq!(e_bd, e_db);

                        val_rr += e_ac
                            * e_bd
                            * int_tilde_q(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_r,
                                delta_rr,
                            );
                        val_lr += e_ca
                            * e_bd
                            * int_tilde_q(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_r,
                                delta_lr,
                            );
                        val_rl += e_ac
                            * e_db
                            * int_tilde_q(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_r,
                                delta_rl,
                            );
                        val_ll += e_ca
                            * e_db
                            * int_tilde_q(
                                (t + tau) as i32,
                                (u + nu) as i32,
                                sigma_l,
                                delta_ll,
                            );

                        assert_abs_diff_eq!(val_rr, val_ll);
                        assert_abs_diff_eq!(val_rr, val_lr);
                        assert_abs_diff_eq!(val_rr, val_rl);
                    }
                }
            }
        }

        assert_abs_diff_eq!(val_rr, val_ll);
        assert_abs_diff_eq!(val_rr, val_lr);
        assert_abs_diff_eq!(val_rr, val_rl);
    }

    #[test]
    fn test_transpose_u() {
        let gaussians = vec![
            G2D::new((0, 0), 1.0, (1.0, 0.0)),
            G2D::new((1, 0), 1.0, (0.0, 0.2)),
            G2D::new((0, 1), 1.0, (0.2, -0.7)),
            G2D::new((1, 1), 1.0, (-1.0, 0.3)),
        ];

        let u = super::construct_coulomb_operator_matrix_elements(&gaussians);
        let l = gaussians.len();

        assert!(u.shape().len() == 4);

        for &dim_len in u.shape().into_iter() {
            assert!(dim_len == l);
        }

        for p in 0..l {
            for q in 0..l {
                for r in 0..l {
                    for s in 0..l {
                        assert_abs_diff_eq!(u[[p, q, r, s]], u[[q, p, s, r]]);
                        assert_abs_diff_eq!(u[[p, q, r, s]], u[[r, q, p, s]]);
                        assert_abs_diff_eq!(u[[p, q, r, s]], u[[p, s, r, q]]);
                        assert_abs_diff_eq!(u[[p, q, r, s]], u[[r, s, p, q]]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_u_symmetries() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..4 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..2), rng.gen_range(0..2)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        for p in 0..gaussians.len() {
            let g_p = &gaussians[p];

            for q in 0..gaussians.len() {
                let g_q = &gaussians[q];

                for r in 0..gaussians.len() {
                    let g_r = &gaussians[r];

                    for s in 0..gaussians.len() {
                        let g_s = &gaussians[s];

                        let u_pqrs =
                            tests::construct_coulomb_operator_matrix_element(
                                g_p, g_q, g_r, g_s,
                            );
                        let u_qpsr =
                            tests::construct_coulomb_operator_matrix_element(
                                g_q, g_p, g_s, g_r,
                            );
                        assert_abs_diff_eq!(u_pqrs, u_qpsr);

                        let u_rqps =
                            tests::construct_coulomb_operator_matrix_element(
                                g_r, g_q, g_p, g_s,
                            );
                        assert_abs_diff_eq!(u_pqrs, u_rqps);

                        let u_psrq =
                            tests::construct_coulomb_operator_matrix_element(
                                g_p, g_s, g_r, g_q,
                            );
                        assert_abs_diff_eq!(u_pqrs, u_psrq);

                        let u_rspq =
                            tests::construct_coulomb_operator_matrix_element(
                                g_r, g_s, g_p, g_q,
                            );
                        assert_abs_diff_eq!(u_pqrs, u_rspq);
                    }
                }
            }
        }
    }

    fn construct_coulomb_operator_matrix_elements(
        gaussians: &Vec<G2D>,
    ) -> Array4<f64> {
        let l = gaussians.len();

        let mut u = Array::zeros((l, l, l, l));
        // let mut counter = 0u64;

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
                        // counter += 1;
                    }
                }
            }
        }
        // dbg!(counter);

        u
    }

    #[test]
    fn compare_coulomb_methods() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..5 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..2), rng.gen_range(0..2)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        let l = gaussians.len();

        let u_test =
            tests::construct_coulomb_operator_matrix_elements(&gaussians);
        let u_perm =
            super::construct_coulomb_operator_matrix_elements(&gaussians);

        for p in 0..l {
            for q in 0..l {
                for r in 0..l {
                    for s in 0..l {
                        assert_abs_diff_eq!(
                            u_test[[p, q, r, s]],
                            u_perm[[p, q, r, s]],
                            epsilon = 1e-15,
                        );
                        assert_abs_diff_eq!(
                            u_test[[p, q, r, s]],
                            u_test[[r, q, p, s]],
                            epsilon = 1e-15,
                        );
                        assert_abs_diff_eq!(
                            u_test[[p, q, r, s]],
                            u_test[[p, s, r, q]],
                            epsilon = 1e-15,
                        );
                        assert_abs_diff_eq!(
                            u_test[[p, q, r, s]],
                            u_test[[r, s, p, q]],
                            epsilon = 1e-15,
                        );
                    }
                }
            }
        }
    }
}
