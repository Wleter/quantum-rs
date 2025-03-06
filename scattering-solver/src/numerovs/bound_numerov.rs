use std::marker::PhantomData;

use faer::{prelude::SolverCore, Mat};
use quantum::{approx_eq, assert_approx_eq, params::particles::Particles};

use crate::{boundary::{Boundary, Direction}, potentials::potential::{MatPotential, SimplePotential}};

use super::{multi_numerov::{MultiNumerovData, MultiRatioNumerovStep}, numerov_modifier::NumerovNodeCount, propagator::{MultiStep, Numerov, PropagatorData, StepAction, StepRule}, single_numerov::{SingleNumerovData, SingleRatioNumerovStep}};

pub struct BoundDiff {
    pub nodes: u64,
    pub diff: f64
}

pub type SingleBoundRatioNumerov<'a, P, S> = BoundRatioNumerov<SingleNumerovData<'a, P>, S, SingleRatioNumerovStep>;
pub type MultiBoundRatioNumerov<'a, P, S> = BoundRatioNumerov<MultiNumerovData<'a, P>, S, MultiRatioNumerovStep>;

pub struct BoundRatioNumerov<D, S, M> 
where
    D: PropagatorData,
    S: StepRule<D>,
    M: MultiStep<D>,
{
    phantom: PhantomData<(D, M)>,
    step_rule: S,
}

impl<D, S, M> BoundRatioNumerov<D, S, M> 
where
    D: PropagatorData,
    S: StepRule<D> + Clone,
    M: MultiStep<D>,
{
    pub fn new(step_rule: S) -> Self {
        Self {
            phantom: PhantomData,
            step_rule,
        }
    }
}

impl<'a, P, S> BoundRatioNumerov<SingleNumerovData<'a, P>, S, SingleRatioNumerovStep>
where
    P: SimplePotential,
    S: StepRule<SingleNumerovData<'a, P>> + Clone,
{
    pub fn bound_diff(&self, potential: &'a P, particles: &Particles, matching: (f64, f64)) -> BoundDiff {
        let boundary = Boundary::new(matching.1, Direction::Inwards, (1e3, 1e5));
        let mut numerov = Numerov::<_, _, SingleRatioNumerovStep>::new(potential, particles, self.step_rule.clone(), boundary);
        while !numerov.data.crossed_distance(matching.0) && numerov.data.psi1 >= 1.0 {
            numerov.variable_step();
        }

        let r_match = numerov.data.r;

        let boundary = Boundary::new(matching.0, Direction::Outwards, (1e3, 1e5));
        let mut numerov = Numerov::<_, _, SingleRatioNumerovStep>::new(potential, particles, self.step_rule.clone(), boundary);
        let mut node_counting_out = NumerovNodeCount::default();
        numerov.propagate_to_with(r_match, &mut node_counting_out);

        let r_match = numerov.data.r;
        let dr_match = numerov.data.dr;
        let psi_match = numerov.data.psi1;

        let r_start = r_match + f64::round((matching.1 - r_match) / dr_match) * dr_match;

        let boundary = Boundary::new(r_start, Direction::Step(-dr_match), (1e3, 1e5));
        let step_rules = MatchPointStepRule::new(self.step_rule.clone(), r_match, dr_match);

        let mut numerov = Numerov::<_, _, SingleRatioNumerovStep>::new(potential, particles, step_rules, boundary);
        let mut node_counting_in = NumerovNodeCount::default();
        numerov.propagate_to_with(r_match - dr_match / 2., &mut node_counting_in);

        assert_approx_eq!(dr_match, numerov.data.dr.abs(), 1e-6);
        assert_approx_eq!(r_match, numerov.data.r + dr_match, 1e-6);

        BoundDiff {
            nodes: node_counting_out.count() + node_counting_in.count(),
            diff: psi_match - 1.0 / numerov.data.psi1,
        }
    }
}

impl<'a, P, S> BoundRatioNumerov<MultiNumerovData<'a, P>, S, MultiRatioNumerovStep> 
where
    P: MatPotential,
    S: StepRule<MultiNumerovData<'a, P>> + Clone,
{
    pub fn bound_diff(&self, potential: &'a P, particles: &'a Particles, matching: (f64, f64)) -> BoundDiff {
        let id = Mat::<f64>::identity(potential.size(), potential.size());

        let r_match = todo!();


        let boundary = Boundary::new(matching.0, Direction::Outwards, (1e3 * &id, 1e5 * &id));
        let mut numerov = Numerov::<_, _, MultiRatioNumerovStep>::new(potential, particles, self.step_rule.clone(), boundary);
        let mut node_counting_out = NumerovNodeCount::default();
        numerov.propagate_to_with(r_match, &mut node_counting_out);

        let r_match = numerov.data.r;
        let dr_match = numerov.data.dr;
        let psi_match = numerov.data.psi1;

        let r_start = r_match + f64::round((matching.1 - r_match) / dr_match) * dr_match;

        let boundary = Boundary::new(r_start, Direction::Step(-dr_match), (1e3 * &id, 1e5 * &id));
        let step_rules = MatchPointStepRule::new(self.step_rule.clone(), r_match, dr_match);

        let mut numerov = Numerov::<_, _, MultiRatioNumerovStep>::new(potential, particles, step_rules, boundary);
        let mut node_counting_in = NumerovNodeCount::default();
        numerov.propagate_to_with(r_match - dr_match / 2., &mut node_counting_in);

        assert_approx_eq!(dr_match, numerov.data.dr.abs(), 1e-6);
        assert_approx_eq!(r_match, numerov.data.r + dr_match, 1e-6);

        BoundDiff {
            nodes: node_counting_out.count() + node_counting_in.count(),
            diff: (psi_match - numerov.data.psi1.partial_piv_lu().inverse()).determinant(),
        }
    }
}


#[derive(Clone)]
pub struct MatchPointStepRule<D, S> 
where
    D: PropagatorData,
    S: StepRule<D>
{
    step_rule: S,
    r_match: f64,
    dr_target: f64,
    _phantom: PhantomData<D>
}

impl<D, S> MatchPointStepRule<D, S>
where 
    D: PropagatorData,
    S: StepRule<D>
{
    pub fn new(step_rule: S, r_match: f64, dr_target: f64) -> Self {
        let dr_target = dr_target.abs();

        Self {
            step_rule,
            r_match,
            dr_target,
            _phantom: PhantomData,
        }
    }
}

impl<'a, P, S> StepRule<SingleNumerovData<'a, P>> for MatchPointStepRule<SingleNumerovData<'a, P>, S>
where 
    S: StepRule<SingleNumerovData<'a, P>>,
    P: SimplePotential
{
    fn get_step(&self, data: &SingleNumerovData<'a, P>) -> f64 {
        self.step_rule.get_step(data)
    }

    fn assign(&mut self, data: &SingleNumerovData<'a, P>) -> StepAction {
        let action = self.step_rule.assign(data);
        let dr_abs = data.dr.abs();

        if is_near(data.dr, data.r, 4. * self.dr_target, self.r_match) &&
            !approx_eq!(dr_abs, self.dr_target, 1e-6) && dr_abs < self.dr_target   
        {
            return StepAction::Double;
        }

        if is_near(data.dr, data.r, 2.0 * self.dr_target, self.r_match) {
            if approx_eq!(dr_abs, self.dr_target, 1e-6) {
                return StepAction::Keep;
            } else if dr_abs > self.dr_target {
                return StepAction::Halve;
            }
        }

        // prevent steps smaller than dr_target
        // so that propagation stops exactly at r_match
        if dr_abs <= self.dr_target || approx_eq!(dr_abs, self.dr_target, 1e-6) {
            match action {
                StepAction::Keep | StepAction::Halve => StepAction::Keep,
                StepAction::Double => StepAction::Double,
            }
        } else {
            action
        }
    }
}

impl<'a, P, S> StepRule<MultiNumerovData<'a, P>> for MatchPointStepRule<MultiNumerovData<'a, P>, S>
where 
    S: StepRule<MultiNumerovData<'a, P>>,
    P: MatPotential
{
    fn get_step(&self, data: &MultiNumerovData<'a, P>) -> f64 {
        self.step_rule.get_step(data)
    }

    fn assign(&mut self, data: &MultiNumerovData<'a, P>) -> StepAction {
        let action = self.step_rule.assign(data);
        let dr_abs = data.dr.abs();

        if is_near(data.dr, data.r, 4. * self.dr_target, self.r_match) &&
            !approx_eq!(dr_abs, self.dr_target, 1e-6) && dr_abs < self.dr_target   
        {
            return StepAction::Double;
        }

        if is_near(data.dr, data.r, self.dr_target, self.r_match) {
            if approx_eq!(dr_abs, self.dr_target, 1e-6) {
                return StepAction::Keep;
            } else if dr_abs > self.dr_target {
                return StepAction::Halve;
            }
        }

        // prevent steps smaller than dr_target
        // so that propagation stops exactly at r_match
        if dr_abs <= self.dr_target || approx_eq!(dr_abs, self.dr_target, 1e-6) {
            match action {
                StepAction::Keep | StepAction::Halve => StepAction::Keep,
                StepAction::Double => StepAction::Double,
            }
        } else {
            action
        }
    }
}

#[inline]
fn is_near(dr: f64, r: f64, dr_target: f64, matching: f64) -> bool {
    (dr > 0. && (r + dr_target >= matching || r + 2.0 * dr >= matching)) 
        || (dr < 0. && (r - dr_target <= matching || r + 2.0 * dr <= matching))
}
