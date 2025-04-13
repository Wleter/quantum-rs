use std::marker::PhantomData;

use faer::Mat;
use quantum::{params::particles::Particles, units::{Au, Energy, EnergyUnit}};
use serde::{Deserialize, Serialize};

use crate::{boundary::{Boundary, Direction}, log_derivatives::{LogDerivative, LogDerivativeReference}, numerovs::StepRule, potentials::potential::MatPotential, propagator::{CoupledEquation, Propagator}};

pub struct BoundMismatch {
    pub nodes: u64,
    pub matching_matrix: Mat<f64>,
    pub matching_eigenvalues: Vec<f64>,
}

pub struct BoundProblemBuilder<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference
{
    particles: Option<&'a Particles>,
    potential: Option<&'a P>,

    step_rule: Option<S>,
    phantom: PhantomData<Prop>,

    r_range: [f64; 3],
}

impl<'a, P, S, Prop> Default for BoundProblemBuilder<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference
{
    fn default() -> Self {
        Self { 
            particles: Default::default(),
            potential: Default::default(),

            step_rule: Default::default(),
            phantom: PhantomData,

            r_range: Default::default(),
        }
    }
}

impl<'a, P, S, Prop> BoundProblemBuilder<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference
{
    pub fn new(particles: &'a Particles, potential: &'a P) -> Self {
        let mut problem = BoundProblemBuilder::default();

        problem.particles = Some(particles);
        problem.potential = Some(potential);

        problem
    }

    pub fn with_equation(mut self, particles: &'a Particles, potential: &'a P) -> Self {
        self.particles = Some(particles);
        self.potential = Some(potential);

        self
    }

    pub fn with_propagation(mut self, step_rule: S, _prop: Prop) -> Self {
        self.step_rule = Some(step_rule);

        self
    }

    pub fn with_range(mut self, r_min: f64, r_match: f64, r_max: f64) -> Self {
        self.r_range = [r_min, r_match, r_max];

        self
    }

    pub fn build(self) -> BoundProblem<'a, P, S, Prop> {
        let particles = self.particles.expect("Did not found particles in BoundBuilder");
        let potential = self.potential.expect("Did not found potential in BoundBuilder");

        let step_rule = self.step_rule.expect("Did not found step rule in BoundBuilder");

        BoundProblem { 
            particles: particles, 
            potential: potential, 
            step_rule: step_rule, 
            phantom: self.phantom, 
            r_min: self.r_range[0], 
            r_match: self.r_range[1], 
            r_max: self.r_range[2]
        }
    }
}

pub struct BoundProblem<'a, P, S, Prop> 
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference
{
    particles: &'a Particles,
    potential: &'a P,

    step_rule: S,
    phantom: PhantomData<Prop>,

    r_min: f64,
    r_match: f64,
    r_max: f64
}

impl<'a, P, S, Prop> BoundProblem<'a, P, S, Prop> 
where
    P: MatPotential,
    S: StepRule<Mat<f64>> + Clone,
    Prop: LogDerivativeReference
{
    pub fn bound_mismatch(&self, energy: Energy<impl EnergyUnit>) -> BoundMismatch {
        let mut particles = self.particles.clone();
        particles.insert(energy.to(Au));

        let eq = CoupledEquation::from_particles(self.potential, &particles);
        let boundary = Boundary::new_exponential_vanishing(self.r_max, &eq);

        let mut propagator = LogDerivative::<_, Prop>::new(eq.clone(), boundary, self.step_rule.clone());
        let solution_in = propagator.propagate_to(self.r_match);

        let boundary = Boundary::new_multi_vanishing(self.r_min, Direction::Outwards, self.potential.size());
        let mut propagator = LogDerivative::<_, Prop>::new(eq, boundary, self.step_rule.clone());
        let solution_out = propagator.propagate_to(self.r_match);

        let matching_matrix = &solution_out.sol.0 - &solution_in.sol.0;
        let nodes = solution_in.nodes + solution_out.nodes;

        let eigenvalues = matching_matrix.self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("could not diagonalize matching matrix");

        let nodes = nodes + eigenvalues.iter().fold(0, |acc, &x| if x < 0. { acc + 1 } else { acc });

        BoundMismatch {
            nodes,
            matching_matrix,
            matching_eigenvalues: eigenvalues
        }
    }

    pub fn bound_states(&self, energy_range: (Energy<impl EnergyUnit>, Energy<impl EnergyUnit>), err: Energy<impl EnergyUnit>) -> BoundStates {
        let lower_mismatch = self.bound_mismatch(energy_range.0);
        let upper_mismatch = self.bound_mismatch(energy_range.1);
        
        let mut bound_energies = vec![];
        let mut bound_nodes = vec![];
        let mut lower_energy = energy_range.0.to(Au);
        let mut upper_energy = energy_range.1.to(Au);

        let mut target_node = upper_mismatch.nodes;
        while target_node > lower_mismatch.nodes {
            let (bound_energy, lower_en) = self.bin_search(target_node, lower_energy, upper_energy, err.to(Au));

            upper_energy = bound_energy;
            lower_energy = lower_en;

            bound_energies.push(bound_energy.to_au());
            bound_nodes.push(target_node);

            target_node -= 1;
        }

        BoundStates { 
            energies: bound_energies, 
            nodes: bound_nodes 
        }
    }

    fn bin_search(&self, target_node: u64, lower: Energy<Au>, upper: Energy<Au>, err: Energy<Au>) -> (Energy<Au>, Energy<Au>) {
        let mut lower = lower.to_au();
        let mut upper = upper.to_au();
        let err = err.to_au();

        let mut before_lower = lower;
        while upper - lower > err {
            let energy_mid = (lower + upper) / 2.;
            let mid_mismatch = self.bound_mismatch(Energy(energy_mid, Au));

            if mid_mismatch.nodes + 1 < target_node {
                before_lower = energy_mid
            }

            if mid_mismatch.nodes >= target_node {
                upper = energy_mid
            } else if mid_mismatch.nodes < target_node {
                lower = energy_mid
            }
        }

        (Energy((lower + upper) / 2., Au), Energy(before_lower, Au))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BoundStates {
    pub energies: Vec<f64>,
    pub nodes: Vec<u64>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BoundStatesDependence {
    pub parameters: Vec<f64>,
    pub bound_states: Vec<BoundStates>,
}

impl BoundStates {
    pub fn with_energy_units(mut self, unit: impl EnergyUnit) -> Self {
        for energy in &mut self.energies {
            *energy = Energy(*energy, Au).to(unit).value()
        }

        self
    }
}