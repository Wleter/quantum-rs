use faer::{Col, Mat, Side, linalg::solvers::Solve};
use scattering_solver::potentials::potential::Potential;

pub struct RKHSInterpolation {
    m: u32,
    alpha_factors: Vec<f64>,
    beta_factors: Vec<f64>,
    distances: Vec<f64>,
}

impl Potential for RKHSInterpolation {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        let result = self
            .alpha_factors
            .iter()
            .zip(self.distances.iter())
            .map(|(alpha, &r_i)| {
                let r_lower = r.min(r_i);
                let r_upper = r.max(r_i);

                alpha / r_upper.powi(self.m as i32 + 1)
                    * self
                        .beta_factors
                        .iter()
                        .enumerate()
                        .map(|(k, beta)| beta * (r_lower / r_upper).powi(k as i32))
                        .sum::<f64>()
            })
            .sum::<f64>();
        *value = result
    }

    fn size(&self) -> usize {
        1
    }
}

impl RKHSInterpolation {
    pub fn new(points: &[f64], values: &[f64]) -> RKHSInterpolation {
        assert!(points.len() == values.len());
        let m = 5;
        let betas = vec![0.05357142857142857, -0.07142857142857142, 0.025];

        let q_matrix = Mat::from_fn(points.len(), points.len(), |i, j| {
            let r_i = points[i];
            let r_j = points[j];

            let r_lower = r_j.min(r_i);
            let r_upper = r_j.max(r_i);

            1. / r_upper.powi(m as i32 + 1)
                * betas
                    .iter()
                    .enumerate()
                    .map(|(k, beta)| beta * (r_lower / r_upper).powi(k as i32))
                    .sum::<f64>()
        });

        let values = Col::from_fn(values.len(), |i| values[i]);

        let cholesky = q_matrix.ldlt(Side::Lower).unwrap();
        let alphas = cholesky.solve(values.as_ref());

        let alphas = alphas.iter().copied().collect();

        RKHSInterpolation {
            m,
            alpha_factors: alphas,
            beta_factors: betas,
            distances: points.to_vec(),
        }
    }
}
