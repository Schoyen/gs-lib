use super::G2D;
use crate::one_dim::OD1D;

pub struct OD2D<'a> {
    pub g_a: &'a G2D,
    pub g_b: &'a G2D,

    pub od_x: OD1D<'a>,
    pub od_y: OD1D<'a>,

    pub tot_exp: f64,
    pub center_diff: (f64, f64),

    pub x_sum_lim: u32,
    pub y_sum_lim: u32,
}

impl<'a> OD2D<'a> {
    pub fn new(g_a: &'a G2D, g_b: &'a G2D) -> Self {
        let od_x = OD1D::new(&g_a.g_x, &g_b.g_x);
        let od_y = OD1D::new(&g_a.g_y, &g_b.g_y);

        assert!(od_x.tot_exp == od_y.tot_exp);

        let tot_exp = od_x.tot_exp;

        let x_sum_lim = od_x.i + od_x.j;
        let y_sum_lim = od_y.i + od_y.j;
        let center_diff = (od_x.center_diff, od_y.center_diff);

        OD2D {
            g_a,
            g_b,
            od_x,
            od_y,
            tot_exp,
            center_diff,
            x_sum_lim,
            y_sum_lim,
        }
    }

    pub fn expansion_coefficients(&mut self, t: i32, u: i32) -> f64 {
        self.od_x.expansion_coefficients(t)
            * self.od_y.expansion_coefficients(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    #[test]
    fn test_construction() {
        let omega = 1.0;

        let gaussians = vec![
            G2D::new((0, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 0), omega / 2.0, (0.0, 0.0)),
            G2D::new((0, 2), omega / 2.0, (0.0, 0.0)),
            G2D::new((1, 1), omega / 2.0, (0.0, 0.0)),
            G2D::new((2, 0), omega / 2.0, (0.0, 0.0)),
        ];

        for a in 0..gaussians.len() {
            let g_a = &gaussians[a];
            let _od_aa = OD2D::new(g_a, g_a);

            for b in 0..gaussians.len() {
                let g_b = &gaussians[b];
                let _od_ab = OD2D::new(g_a, g_b);
                let _od_ba = OD2D::new(g_b, g_a);
            }
        }

        assert!(true);
    }

    #[test]
    fn test_expansion_coefficients() {
        let g_00 = G2D::new((0, 0), 1.0, (1.0, 2.0));
        let g_21 = G2D::new((2, 1), 0.7, (-0.3, 0.0));

        let mut od_00_00 = OD2D::new(&g_00, &g_00);
        let mut od_00_21 = OD2D::new(&g_00, &g_21);
        let mut od_21_00 = OD2D::new(&g_21, &g_00);
        let mut od_21_21 = OD2D::new(&g_21, &g_21);

        assert_abs_diff_eq!(od_00_00.expansion_coefficients(0, 0), 1.0);
        assert_abs_diff_eq!(od_00_00.expansion_coefficients(0, 1), 0.0);
        assert_abs_diff_eq!(od_00_00.expansion_coefficients(1, 0), 0.0);

        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(0, 0),
            0.0993094379070067
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(0, 0),
            od_21_00.expansion_coefficients(0, 0)
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(0, 1),
            0.024827359476751677
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(0, 1),
            od_21_00.expansion_coefficients(0, 1)
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(1, 0),
            0.05082766507051525
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(1, 0),
            od_21_00.expansion_coefficients(1, 0)
        );
        assert_abs_diff_eq!(od_00_21.expansion_coefficients(0, 2), 0.0);
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(0, 2),
            od_21_00.expansion_coefficients(0, 2)
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(1, 1),
            0.012706916267628812
        );
        assert_abs_diff_eq!(
            od_00_21.expansion_coefficients(1, 1),
            od_21_00.expansion_coefficients(1, 1)
        );

        assert_abs_diff_eq!(
            od_21_21.expansion_coefficients(0, 0),
            0.13666180758017493
        );
        assert_abs_diff_eq!(od_21_21.expansion_coefficients(0, 1), 0.0);
        assert_abs_diff_eq!(od_21_21.expansion_coefficients(1, 0), 0.0);
        assert_abs_diff_eq!(
            od_21_21.expansion_coefficients(0, 2),
            0.04880778842149105
        );
        assert_abs_diff_eq!(
            od_21_21.expansion_coefficients(2, 0),
            0.0976155768429821
        );
    }

    #[test]
    fn test_symmetry_of_expansion_coefficients() {
        let mut rng = rand::thread_rng();
        let mut gaussians = Vec::new();

        for _i in 0..6 {
            gaussians.push(G2D::new(
                (rng.gen_range(0..3), rng.gen_range(0..3)),
                0.5 + rng.gen::<f64>(),
                (
                    2.0 * (rng.gen::<f64>() - 0.5),
                    2.0 * (rng.gen::<f64>() - 0.5),
                ),
            ));
        }

        for i in 0..gaussians.len() {
            let g_i = &gaussians[i];

            for j in i..gaussians.len() {
                let g_j = &gaussians[j];
                let mut od_ij = OD2D::new(g_i, g_j);
                let mut od_ji = OD2D::new(g_j, g_i);

                assert!(od_ij.x_sum_lim == od_ji.x_sum_lim);
                assert!(od_ij.y_sum_lim == od_ji.y_sum_lim);

                for t in 0..(od_ij.x_sum_lim + 1) {
                    for u in 0..(od_ij.y_sum_lim + 1) {
                        let e_tu_ij =
                            od_ij.expansion_coefficients(t as i32, u as i32);
                        let e_ut_ij =
                            od_ij.expansion_coefficients(u as i32, t as i32);
                        let e_tu_ji =
                            od_ji.expansion_coefficients(t as i32, u as i32);
                        let e_ut_ji =
                            od_ji.expansion_coefficients(u as i32, t as i32);

                        assert_abs_diff_eq!(e_tu_ij, e_tu_ji);
                        assert_abs_diff_eq!(e_ut_ij, e_ut_ji);
                    }
                }
            }
        }
    }
}
