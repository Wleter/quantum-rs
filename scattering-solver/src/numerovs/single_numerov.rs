use std::marker::PhantomData;

use num::complex::Complex64;
use quantum::utility::{riccati_j, riccati_n};

use crate::{
    boundary::{Boundary, Direction},
    observables::s_matrix::SMatrix,
    propagator::{Equation, MultiStep, Propagator, Repr, Solution},
};

use super::{Numerov, Ratio, StepAction, StepRule, propagator_watcher::PropagatorWatcher};

/// doi: 10.1063/1.435384
pub type SingleRNumerov<'a, S> = Numerov<'a, f64, Ratio<f64>, SingleRNumerovStep, S>;

impl<'a, S: StepRule<f64>> SingleRNumerov<'a, S> {
    pub fn new(mut eq: Equation<'a, f64>, boundary: Boundary<f64>, step_rule: S) -> Self {
        let r = boundary.r_start;

        eq.buffer_w_matrix(r);
        let dr = match boundary.direction {
            Direction::Inwards => -step_rule.get_step(&eq.buffered_w_matrix()),
            Direction::Outwards => step_rule.get_step(&eq.buffered_w_matrix()),
            Direction::Step(dr) => dr,
        };

        let sol = Solution {
            r,
            dr,
            sol: Ratio(boundary.start_value),
            nodes: 0
        };

        let f3 = 1. + dr * dr / 12. * eq.w_matrix(r - 2. * dr);
        let f2 = 1. + dr * dr / 12. * eq.w_matrix(r - dr);
        let f1 = 1. + dr * dr / 12. * eq.buffered_w_matrix();

        let sol_last = Ratio(boundary.before_value);

        let multi_step = SingleRNumerovStep {
            f1,
            f2,
            f3,
            sol_last,
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

impl<R, M, S> Propagator<f64, R> for Numerov<'_, f64, R, M, S>
where
    R: Repr<f64>,
    M: MultiStep<f64, R>,
    S: StepRule<f64>,
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
        modifier: &mut impl PropagatorWatcher<f64, R>,
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
        self.equation
            .buffer_w_matrix(self.solution.r + self.solution.dr);

        let mut action = self
            .step_rule
            .step_action(self.solution.dr, &self.equation.buffered_w_matrix());

        if let StepAction::Double = action {
            self.multi_step
                .double_the_step(&mut self.solution, &self.equation);

            self.equation
                .buffer_w_matrix(self.solution.r + self.solution.dr);
        }

        let mut halved = false;
        while let StepAction::Halve = action {
            self.multi_step
                .halve_the_step(&mut self.solution, &self.equation);
            action = self
                .step_rule
                .step_action(self.solution.dr, &self.equation.buffered_w_matrix());
            halved = true;
        }

        if halved {
            self.equation
                .buffer_w_matrix(self.solution.r + self.solution.dr);
        }

        self.multi_step
            .perform_step(&mut self.solution, &self.equation);

        &self.solution
    }
}

#[derive(Clone, Debug, Default)]
pub struct SingleRNumerovStep {
    f1: f64,
    f2: f64,
    f3: f64,

    sol_last: Ratio<f64>,
}

impl MultiStep<f64, Ratio<f64>> for SingleRNumerovStep {
    /// Performs the step of the propagation using buffered w_matrix value
    fn perform_step(&mut self, sol: &mut Solution<Ratio<f64>>, eq: &Equation<f64>) {
        sol.r += sol.dr;

        let f = 1.0 + sol.dr * sol.dr * eq.buffered_w_matrix() / 12.0;
        let sol_new = (12.0 - 10.0 * self.f1 - self.f2 / sol.sol.0) / f;

        self.f3 = self.f2;
        self.f2 = self.f1;
        self.f1 = f;

        if sol_new < 0. {
            sol.nodes += 1
        }

        self.sol_last = sol.sol;
        sol.sol.0 = sol_new;
    }

    fn halve_the_step(&mut self, sol: &mut Solution<Ratio<f64>>, eq: &Equation<f64>) {
        sol.dr /= 2.;

        let f = 1.0 + sol.dr * sol.dr * eq.w_matrix(sol.r - sol.dr) / 12.0;
        self.f1 = self.f1 / 4.0 + 0.75;
        self.f2 = self.f2 / 4.0 + 0.75;

        let sol_half = (self.f1 * sol.sol.0 + self.f2) / (12.0 - 10.0 * f);
        self.f2 = f;

        self.sol_last.0 = sol_half;
        sol.sol.0 /= sol_half;
    }

    fn double_the_step(&mut self, sol: &mut Solution<Ratio<f64>>, _eq: &Equation<f64>) {
        sol.dr *= 2.0;

        self.f2 = 4.0 * self.f3 - 3.0;
        self.f1 = 4.0 * self.f1 - 3.0;

        sol.sol.0 *= self.sol_last.0;
    }
}

impl Solution<Ratio<f64>> {
    pub fn s_matrix(&self, eq: &Equation<f64>) -> SMatrix {
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;
        let wave_ratio = self.sol.0;

        let mut asymptotic = 0.0;
        eq.potential.value_inplace(r_last, &mut asymptotic);

        let momentum = (2.0 * eq.mass * (eq.energy - asymptotic)).sqrt();
        if momentum.is_nan() {
            panic!("propagated in closed channel");
        }

        assert!(eq.asymptotic.centrifugal.len() == 1);

        let j_last = riccati_j(eq.asymptotic.centrifugal[0].0, momentum * r_last) / momentum.sqrt();
        let j_prev_last =
            riccati_j(eq.asymptotic.centrifugal[0].0, momentum * r_prev_last) / momentum.sqrt();
        let n_last = riccati_n(eq.asymptotic.centrifugal[0].0, momentum * r_last) / momentum.sqrt();
        let n_prev_last =
            riccati_n(eq.asymptotic.centrifugal[0].0, momentum * r_prev_last) / momentum.sqrt();

        let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);

        let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        SMatrix::new_single(s_matrix, momentum)
    }
}

#[cfg(test)]
mod test {
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
        boundary::{Boundary, Direction},
        numerovs::{LocalWavelengthStepRule, single_numerov::SingleRNumerov},
        potentials::{potential::SimplePotential, potential_factory::create_lj},
        propagator::{Propagator, SingleEquation},
    };

    fn potential() -> impl SimplePotential {
        create_lj(Energy(0.002, Au), Distance(9., Au))
    }

    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        Particles::new_pair(particle1, particle2, energy)
    }

    #[test]
    fn test_numerov() {
        let particles = particles();
        let potential = potential();

        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001, 1.002));

        let eq = SingleEquation::from_particles(&potential, &particles);

        let mut numerov = SingleRNumerov::new(eq, boundary, LocalWavelengthStepRule::default());

        assert_approx_eq!(numerov.solution.dr, 4.336507e-4, 1e-6);

        assert_approx_eq!(numerov.solution.sol.0, 1.001, 1e-6);
        assert_approx_eq!(numerov.multi_step.sol_last.0, 1.002, 1e-6);

        numerov.step();
        assert_approx_eq!(numerov.solution.sol.0, 1.0011569, 1e-6);

        numerov.step();
        numerov.step();
        numerov.step();
        assert_approx_eq!(numerov.solution.sol.0, 1.00162454, 1e-6);
    }

    #[test]
    fn test_scattering() {
        let particles = particles();
        let potential = potential();

        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001, 1.002));
        let eq = SingleEquation::from_particles(&potential, &particles);

        let mut numerov = SingleRNumerov::new(eq, boundary, LocalWavelengthStepRule::default());

        numerov.propagate_to(1500.0);
        let s_matrix = numerov.s_matrix();

        // values at which the result was correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -15.51539, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -1.1120368e-12, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 3025.06779, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 1.03508256e-23, 1e-6);
    }
}
