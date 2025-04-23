use std::{marker::PhantomData, mem::swap};

use faer::{
    Accum, Mat, Par,
    dyn_stack::MemBuffer,
    linalg::{matmul::matmul, solvers::DenseSolveCore},
    prelude::c64,
    unzip, zip,
};
use quantum::utility::{ratio_riccati_i, ratio_riccati_k, riccati_j, riccati_n};

use crate::{
    boundary::Boundary,
    observables::s_matrix::SMatrix,
    propagator::{Equation, MultiStep, Propagator, Repr, Solution},
    utility::{get_ldlt_inverse_buffer, inverse_ldlt_inplace, inverse_ldlt_inplace_nodes},
};

use super::{Numerov, Ratio, StepAction, StepRule, propagator_watcher::PropagatorWatcher};

/// 10.1063/1.436421
pub type MultiRNumerov<'a, S> = Numerov<'a, Mat<f64>, Ratio<Mat<f64>>, MultiRNumerovStep, S>;

impl<'a, S: StepRule<Mat<f64>>> MultiRNumerov<'a, S> {
    pub fn new(mut eq: Equation<'a, Mat<f64>>, boundary: Boundary<Mat<f64>>, step_rule: S) -> Self {
        let size = eq.potential.size();

        let r = boundary.r_start;

        eq.buffer_w_matrix(r);
        let dr = match boundary.direction {
            crate::boundary::Direction::Inwards => -step_rule.get_step(eq.buffered_w_matrix()),
            crate::boundary::Direction::Outwards => step_rule.get_step(eq.buffered_w_matrix()),
            crate::boundary::Direction::Step(dr) => dr,
        };

        let mut f_last = Mat::zeros(size, size);
        eq.w_matrix(r - dr, &mut f_last);

        let mut f_prev_last = Mat::zeros(size, size);
        eq.w_matrix(r - 2. * dr, &mut f_prev_last);

        let f_last = eq.unit.as_ref() + dr * dr / 12. * f_last;
        let f_prev_last = eq.unit.as_ref() + dr * dr / 12. * f_prev_last;
        let f = eq.unit.as_ref() + dr * dr / 12. * eq.buffered_w_matrix();

        let sol = Solution {
            r,
            dr,
            sol: Ratio(&f * boundary.start_value * f_last.partial_piv_lu().inverse()),
            nodes: 0,
        };

        let sol_last =
            Ratio(&f_last * boundary.before_value * f_prev_last.partial_piv_lu().inverse());

        let multi_step = MultiRNumerovStep {
            sol_last,
            f,
            f_last,
            f_prev_last,
            buffer1: Mat::zeros(size, size),
            buffer2: Mat::zeros(size, size),
            buffer3: Mat::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),
        };

        Self {
            equation: eq,
            solution: sol,
            multi_step,
            step_rule,
            phantom: PhantomData,
        }
    }

    pub fn s_matrix(&self) -> SMatrix {
        self.solution.s_matrix(&self.equation)
    }
}

impl<R, M, S> Propagator<Mat<f64>, R> for Numerov<'_, Mat<f64>, R, M, S>
where
    R: Repr<Mat<f64>>,
    M: MultiStep<Mat<f64>, R>,
    S: StepRule<Mat<f64>>,
{
    fn propagate_to(&mut self, r: f64) -> &Solution<R> {
        while (self.solution.r - r) * self.solution.dr.signum() <= 0. {
            self.step();
        }

        &self.solution
    }

    fn propagate_to_with(
        &mut self,
        r: f64,
        modifier: &mut impl PropagatorWatcher<Mat<f64>, R>,
    ) -> &Solution<R> {
        modifier.before(&self.solution, &self.equation, r);

        while (self.solution.r - r) * self.solution.dr.signum() <= 0. {
            self.step();
            modifier.after_step(&self.solution, &self.equation);
        }

        modifier.after_prop(&self.solution, &self.equation);

        &self.solution
    }

    fn step(&mut self) -> &Solution<R> {
        self.equation.buffer_w_matrix(self.solution.r);

        let mut action = self
            .step_rule
            .step_action(self.solution.dr, self.equation.buffered_w_matrix());

        if let StepAction::Double = action {
            self.multi_step
                .double_the_step(&mut self.solution, &self.equation);
        }

        while let StepAction::Halve = action {
            self.multi_step
                .halve_the_step(&mut self.solution, &self.equation);

            action = self
                .step_rule
                .step_action(self.solution.dr, self.equation.buffered_w_matrix());
        }

        self.multi_step
            .perform_step(&mut self.solution, &self.equation);

        &self.solution
    }
}

pub struct MultiRNumerovStep {
    sol_last: Ratio<Mat<f64>>,

    f: Mat<f64>,
    f_last: Mat<f64>,
    f_prev_last: Mat<f64>,

    buffer1: Mat<f64>,
    buffer2: Mat<f64>,
    buffer3: Mat<f64>,
    inverse_buffer: MemBuffer,
}

impl MultiStep<Mat<f64>, Ratio<Mat<f64>>> for MultiRNumerovStep {
    /// Performs the step of the propagation using buffered w_matrix value
    fn perform_step(&mut self, sol: &mut Solution<Ratio<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        sol.r += sol.dr;

        zip!(
            self.buffer1.as_mut(),
            eq.unit.as_ref(),
            eq.buffered_w_matrix().as_ref()
        )
        .for_each(|unzip!(b1, u, c)| *b1 = u + sol.dr * sol.dr / 12. * c);
        // buffer1 is (1 - T_n)

        let artificial = inverse_ldlt_inplace_nodes(
            self.buffer1.as_ref(),
            self.sol_last.0.as_mut(),
            &mut self.inverse_buffer,
        );
        // sol_last is (1 - T_n)^-1

        zip!(
            self.buffer3.as_mut(),
            eq.unit.as_ref(),
            self.buffer1.as_ref()
        )
        .for_each(|unzip!(b3, u, b1)| *b3 = 12. * u - 10. * *b1);
        // buffer3 is (2 + 10T_n)

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            self.buffer3.as_ref(),
            self.sol_last.0.as_ref(),
            1.,
            Par::Seq,
        );
        // buffer2 is U_n

        let nodes = inverse_ldlt_inplace_nodes(
            sol.sol.0.as_ref(),
            self.sol_last.0.as_mut(),
            &mut self.inverse_buffer,
        );

        if nodes >= artificial {
            println!("artificial nodes is higher than nodes")
        }
        // assert!(nodes >= artificial);
        // sol.nodes += nodes - artificial;
        // if artificial > 0 {
        //     println!("r: {} {} {} {}", sol.r, sol.nodes, nodes, artificial);
        //     panic!("testing panic")
        // }

        zip!(self.sol_last.0.as_mut(), self.buffer2.as_ref())
            .for_each(|unzip!(sol, u)| *sol = u - *sol);
        // sol_last is R_n

        swap(&mut self.sol_last, &mut sol.sol);

        swap(&mut self.f_prev_last, &mut self.f_last);
        swap(&mut self.f_last, &mut self.f);
        swap(&mut self.f, &mut self.buffer1);
    }

    fn halve_the_step(&mut self, sol: &mut Solution<Ratio<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        sol.dr /= 2.0;

        inverse_ldlt_inplace(
            self.f.as_ref(),
            self.buffer1.as_mut(),
            &mut self.inverse_buffer,
        );

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            sol.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            sol.sol.0.as_mut(),
            Accum::Replace,
            self.buffer2.as_ref(),
            self.f_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), eq.unit.as_ref()).for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        zip!(self.f_last.as_mut(), eq.unit.as_ref())
            .for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            sol.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(
            self.f_last.as_ref(),
            self.buffer2.as_mut(),
            &mut self.inverse_buffer,
        );

        matmul(
            sol.sol.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );

        ///////////////////////////////////////////////////////

        eq.w_matrix(sol.r - sol.dr, &mut self.buffer2);
        zip!(
            self.f_prev_last.as_mut(),
            eq.unit.as_ref(),
            self.buffer2.as_ref()
        )
        .for_each(|unzip!(b1, u, c)| *b1 = u + sol.dr * sol.dr / 12. * c);
        // f_prev_last is (1 - T_n)

        inverse_ldlt_inplace(
            self.f_prev_last.as_ref(),
            self.buffer3.as_mut(),
            &mut self.inverse_buffer,
        );
        // buffer3 is (1 - T_n)^-1

        zip!(
            self.buffer1.as_mut(),
            eq.unit.as_ref(),
            self.f_prev_last.as_ref()
        )
        .for_each(|unzip!(b1, u, f)| *b1 = 12. * u - 10. * f);
        // buffer1 is (2 + 10T_n)

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer3.as_ref(),
            1.,
            Par::Seq,
        );
        // buffer2 is U_n

        inverse_ldlt_inplace(
            self.buffer2.as_ref(),
            self.buffer1.as_mut(),
            &mut self.inverse_buffer,
        );
        // buffer1 is U_n^-1

        zip!(self.buffer2.as_mut(), sol.sol.0.as_ref(), eq.unit.as_ref())
            .for_each(|unzip!(b2, sol, u)| *b2 = sol + u);

        matmul(
            self.sol_last.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(
            self.sol_last.0.as_ref(),
            self.buffer1.as_mut(),
            &mut self.inverse_buffer,
        );

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            sol.sol.0.as_ref(),
            self.buffer1.as_ref(),
            1.,
            Par::Seq,
        );

        swap(&mut sol.sol.0, &mut self.buffer2);
        swap(&mut self.f_prev_last, &mut self.f_last)
    }

    fn double_the_step(&mut self, sol: &mut Solution<Ratio<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        sol.dr *= 2.;

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            sol.sol.0.as_ref(),
            self.sol_last.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(
            self.f.as_ref(),
            self.buffer2.as_mut(),
            &mut self.inverse_buffer,
        );

        matmul(
            self.buffer3.as_mut(),
            Accum::Replace,
            self.buffer2.as_ref(),
            self.buffer1.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            sol.sol.0.as_mut(),
            Accum::Replace,
            self.buffer3.as_ref(),
            self.f_prev_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), eq.unit.as_ref()).for_each(|unzip!(f, u)| *f = 4. * *f - 3. * u);

        zip!(
            self.f_last.as_mut(),
            eq.unit.as_ref(),
            self.f_prev_last.as_ref()
        )
        .for_each(|unzip!(f, u, f_prev)| *f = 4. * *f_prev - 3. * u);

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            sol.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(
            self.f_last.as_ref(),
            self.buffer2.as_mut(),
            &mut self.inverse_buffer,
        );

        matmul(
            sol.sol.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );
    }
}

impl Solution<Ratio<Mat<f64>>> {
    pub fn s_matrix(&self, eq: &Equation<Mat<f64>>) -> SMatrix {
        let size = eq.potential.size();
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;

        let mut f_last = Mat::zeros(size, size);
        eq.w_matrix(r_last, &mut f_last);
        f_last *= self.dr * self.dr / 12.;
        f_last += &eq.unit;

        let mut f_prev_last = Mat::zeros(size, size);
        eq.w_matrix(r_prev_last, &mut f_prev_last);
        f_prev_last *= self.dr * self.dr / 12.;
        f_prev_last += &eq.unit;

        let wave_ratio = f_last.partial_piv_lu().inverse() * self.sol.0.as_ref() * f_prev_last;

        let asymptotic = &eq.asymptotic(r_last);

        let is_open_channel = asymptotic
            .iter()
            .map(|&val| val < eq.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .iter()
            .map(|&val| (2.0 * eq.mass * (eq.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = Mat::zeros(size, size);
        let mut j_prev_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_prev_last = Mat::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            let l = eq.asymptotic.centrifugal[i].0;
            if is_open_channel[i] {
                j_last[(i, i)] = riccati_j(l, momentum * r_last) / momentum.sqrt();
                j_prev_last[(i, i)] = riccati_j(l, momentum * r_prev_last) / momentum.sqrt();
                n_last[(i, i)] = riccati_n(l, momentum * r_last) / momentum.sqrt();
                n_prev_last[(i, i)] = riccati_n(l, momentum * r_prev_last) / momentum.sqrt();
            } else {
                j_last[(i, i)] = ratio_riccati_i(l, momentum * r_last, momentum * r_prev_last);
                j_prev_last[(i, i)] = 1.0;
                n_last[(i, i)] = ratio_riccati_k(l, momentum * r_last, momentum * r_prev_last);
                n_prev_last[(i, i)] = 1.0;
            }
        }

        let denominator = (&wave_ratio * n_prev_last - n_last).partial_piv_lu();
        let denominator = denominator.inverse();

        let k_matrix = -denominator * (wave_ratio * j_prev_last - j_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while !is_open_channel[i_full] {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while !is_open_channel[j_full] {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = c64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = Mat::<c64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).partial_piv_lu();
        let denominator = denominator.inverse();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == eq.asymptotic.entrance)
            .expect("Closed entrance channel")
            .0;

        SMatrix::new(s_matrix, momenta[eq.asymptotic.entrance], entrance)
    }
}

#[cfg(test)]
mod test {
    use faer::Mat;
    use quantum::{
        assert_approx_eq,
        params::{particle_factory::create_atom, particles::Particles},
        units::{
            Au,
            distance_units::Distance,
            energy_units::{Energy, Kelvin},
        },
    };

    use crate::{
        boundary::{Asymptotic, Boundary, Direction},
        numerovs::{LocalWavelengthStepRule, multi_numerov::MultiRNumerov},
        potentials::{
            dispersion_potential::Dispersion,
            gaussian_coupling::GaussianCoupling,
            multi_coupling::MultiCoupling,
            multi_diag_potential::Diagonal,
            pair_potential::PairPotential,
            potential::{MatPotential, Potential},
            potential_factory::create_lj,
        },
        propagator::{CoupledEquation, Propagator},
        utility::AngMomentum,
    };

    pub fn potential() -> impl MatPotential {
        let potential_lj1 = create_lj(Energy(0.002, Au), Distance(9., Au));
        let mut potential_lj2 = create_lj(Energy(0.0021, Au), Distance(8.9, Au));
        potential_lj2.add_potential(Dispersion::new(Energy(1., Kelvin).to_au(), 0));

        let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);

        let potential = Diagonal::<Mat<f64>, _>::from_vec(vec![potential_lj1, potential_lj2]);
        let coupling = MultiCoupling::<Mat<f64>, _>::new_neighboring(vec![coupling]);

        PairPotential::new(potential, coupling)
    }

    pub fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(Asymptotic {
            centrifugal: vec![AngMomentum(0); 2],
            entrance: 0,
            channel_energies: vec![0., Energy(0.0021, Kelvin).to_au()],
            channel_states: Mat::identity(2, 2),
        });

        particles
    }

    #[test]
    fn test_numerov() {
        let particles = particles();
        let potential = potential();

        let id = Mat::<f64>::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let eq = CoupledEquation::from_particles(&potential, &particles);

        let mut numerov = MultiRNumerov::new(eq, boundary, LocalWavelengthStepRule::default());

        assert_approx_eq!(numerov.solution.dr, 4.58881145e-4, 1e-6);

        assert_approx_eq!(numerov.solution.sol.0[(0, 0)], 1.001, 1e-6);
        assert_approx_eq!(numerov.multi_step.sol_last.0[(0, 0)], 1.002, 1e-6);

        numerov.step();

        assert_approx_eq!(numerov.solution.sol.0[(0, 0)], 1.001175827, 1e-6);
        assert_approx_eq!(numerov.solution.sol.0[(1, 0)], 6.26398e-9, 1e-6);

        numerov.step();
        numerov.step();
        numerov.step();

        assert_approx_eq!(numerov.solution.sol.0[(0, 0)], 1.0016997, 1e-6);
        assert_approx_eq!(numerov.solution.sol.0[(1, 0)], 2.497223e-8, 1e-6);
    }

    #[test]
    fn test_scattering() {
        let particles = particles();
        let potential = potential();

        let boundary = Boundary::new_multi_vanishing(6.5, Direction::Outwards, potential.size());
        let eq = CoupledEquation::from_particles(&potential, &particles);

        let mut numerov = MultiRNumerov::new(eq, boundary, LocalWavelengthStepRule::default());

        numerov.propagate_to(1500.0);
        let s_matrix = numerov.s_matrix();

        // values at which the result was correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -37.07176, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -1.550004e-12, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 1.7270067e4, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 4.1425318e-23, 1e-6);
    }
}
