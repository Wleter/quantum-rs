use std::marker::PhantomData;

use faer::Mat;
use quantum::{
    params::particles::Particles,
    units::{Au, Energy, EnergyUnit},
};
use serde::{Deserialize, Serialize};

use crate::{
    boundary::{Boundary, Direction},
    log_derivatives::{LogDerivative, LogDerivativeReference, WaveLogDerivStorage},
    numerovs::StepRule,
    potentials::potential::MatPotential,
    propagator::{CoupledEquation, Propagator},
    utility::brent_root_method,
};

#[derive(Clone, Debug)]
pub struct BoundMismatch {
    pub nodes: u64,
    pub matching_matrix: Mat<f64>,
    pub matching_eigenvalues: Vec<f64>,
    pub energy: Energy<Au>,
}

pub struct BoundProblemBuilder<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
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
    Prop: LogDerivativeReference,
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
    Prop: LogDerivativeReference,
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
        let particles = self
            .particles
            .expect("Did not found particles in BoundBuilder");
        let potential = self
            .potential
            .expect("Did not found potential in BoundBuilder");

        let step_rule = self
            .step_rule
            .expect("Did not found step rule in BoundBuilder");

        BoundProblem {
            particles: particles,
            potential: potential,
            step_rule: step_rule,
            phantom: self.phantom,
            r_min: self.r_range[0],
            r_match: self.r_range[1],
            r_max: self.r_range[2],
        }
    }
}

pub struct BoundProblem<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
{
    particles: &'a Particles,
    potential: &'a P,

    step_rule: S,
    phantom: PhantomData<Prop>,

    r_min: f64,
    r_match: f64,
    r_max: f64,
}

impl<'a, P, S, Prop> BoundProblem<'a, P, S, Prop>
where
    P: MatPotential,
    S: StepRule<Mat<f64>> + Clone,
    Prop: LogDerivativeReference,
{
    pub fn bound_mismatch(&self, energy: Energy<impl EnergyUnit>) -> BoundMismatch {
        let mut particles = self.particles.clone();
        particles.insert(energy.to(Au));

        let eq = CoupledEquation::from_particles(self.potential, &particles);
        let boundary = Boundary::new_exponential_vanishing(self.r_max, &eq);

        let mut propagator =
            LogDerivative::<_, Prop>::new(eq.clone(), boundary, self.step_rule.clone());
        let solution_in = propagator.propagate_to(self.r_match);

        let boundary =
            Boundary::new_multi_vanishing(self.r_min, Direction::Outwards, self.potential.size());
        let mut propagator = LogDerivative::<_, Prop>::new(eq, boundary, self.step_rule.clone());
        let solution_out = propagator.propagate_to(self.r_match);

        let matching_matrix = &solution_out.sol.0 - &solution_in.sol.0;
        let nodes = solution_in.nodes + solution_out.nodes;

        let eigenvalues = matching_matrix
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("could not diagonalize matching matrix");

        let nodes = nodes
            + eigenvalues
                .iter()
                .fold(0, |acc, &x| if x < 0. { acc + 1 } else { acc });

        BoundMismatch {
            nodes,
            matching_matrix,
            matching_eigenvalues: eigenvalues,
            energy: energy.to(Au),
        }
    }

    pub fn bound_states(
        &self,
        energy_range: (Energy<impl EnergyUnit>, Energy<impl EnergyUnit>),
        err: Energy<impl EnergyUnit>,
    ) -> BoundStates {
        let lower_mismatch = self.bound_mismatch(energy_range.0);
        let upper_mismatch = self.bound_mismatch(energy_range.1);
        let err = err.to(Au);

        let mut bound_energies = vec![];
        let mut bound_nodes = vec![];

        let upper_node = upper_mismatch.nodes;
        let lower_node = lower_mismatch.nodes;
        let states_no = (upper_node - lower_node) as usize;

        let mut mismatch_node = vec![None; states_no + 1];
        mismatch_node[0] = Some(lower_mismatch);
        mismatch_node[states_no] = Some(upper_mismatch);

        if upper_node == 0 {
            return BoundStates {
                energies: vec![],
                nodes: vec![],
                occupations: None,
            }
        }
        let mut target_nodes = upper_node - 1;
        while target_nodes >= lower_node {
            let bound_energy = self.search_state(lower_node, target_nodes, &mut mismatch_node, err);

            bound_energies.push(bound_energy.to_au());
            bound_nodes.push(target_nodes);

            if target_nodes == 0 {
                break
            }
            target_nodes -= 1;
        }

        BoundStates {
            energies: bound_energies,
            nodes: bound_nodes,
            occupations: None
        }
    }

    pub fn bound_waves(&self, bounds: &BoundStates) -> impl Iterator<Item = WaveFunction> {
        bounds.energies.iter()
            .map(move |&e| {
                let mut particles = self.particles.clone();
                particles.insert(Energy(e, Au));

                let eq = CoupledEquation::from_particles(self.potential, &particles);
                let boundary = Boundary::new_exponential_vanishing(self.r_max, &eq);

                let mut propagator_in = LogDerivative::<_, Prop>::new(eq.clone(), boundary, self.step_rule.clone());
                propagator_in.with_storage(WaveLogDerivStorage::new(true));
                let solution_in = propagator_in.propagate_to(self.r_match);

                let boundary = Boundary::new_multi_vanishing(self.r_min, Direction::Outwards, self.potential.size());
                let mut propagator_out = LogDerivative::<_, Prop>::new(eq, boundary, self.step_rule.clone());
                propagator_out.with_storage(WaveLogDerivStorage::new(true));
                let solution_out = propagator_out.propagate_to(self.r_match);

                let matching_matrix = &solution_out.sol.0 - &solution_in.sol.0;

                let eigen = matching_matrix
                    .self_adjoint_eigen(faer::Side::Lower)
                    .expect("could not diagonalize matching matrix");

                let index = eigen.S().column_vector().iter()
                    .enumerate()
                    .min_by(|x, y| x.1.abs().partial_cmp(&y.1.abs()).unwrap())
                    .unwrap().0;

                let init_wave = eigen.U().col(index);

                let wave_in = propagator_in.storage().as_ref().unwrap().reconstruct(init_wave);
                let mut wave_out = propagator_out.storage().as_ref().unwrap().reconstruct(init_wave);

                wave_out.reverse();
                wave_out.extend(wave_in);

                wave_out.normalize()
            })
    }

    fn search_state(
        &self,
        index_offset: u64,
        target_nodes: u64,
        mismatch_node: &mut Vec<Option<BoundMismatch>>,
        err: Energy<Au>,
    ) -> Energy<Au> {
        let err = err.to_au();

        let node_index = (target_nodes - index_offset) as usize;

        let mut lower_bound = mismatch_node
            .iter()
            .take(node_index + 1)
            .filter(|&x| x.is_some())
            .last()
            .expect("Expected at least one some element in energy node")
            .as_ref()
            .unwrap()
            .clone();

        let mut upper_bound = mismatch_node
            .iter()
            .skip(node_index + 1)
            .filter(|&x| x.is_some())
            .next()
            .expect("Expected at least one some element in energy node")
            .as_ref()
            .unwrap()
            .clone();

        let index = lower_bound
            .matching_eigenvalues
            .partition_point(|&x| x < 0.);
        let mut lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);

        let index = upper_bound
            .matching_eigenvalues
            .partition_point(|&x| x < 0.);
        let mut upper_eigenvalue = upper_bound.matching_eigenvalues.get(index - 1);

        while upper_bound.nodes != target_nodes + 1
            || lower_bound.nodes != target_nodes
            || lower_eigenvalue.is_none()
            || upper_eigenvalue.is_none()
        {
            let energy_mid = (upper_bound.energy.to_au() + lower_bound.energy.to_au()) / 2.;
            if upper_bound.energy.to_au() - lower_bound.energy.to_au() < err {
                return Energy(energy_mid, Au);
            }

            let mid_mismatch = self.bound_mismatch(Energy(energy_mid, Au));

            let index = (mid_mismatch.nodes - index_offset) as usize;

            if mismatch_node[index].is_none() {
                mismatch_node[index] = Some(mid_mismatch.clone());
            }

            if mid_mismatch.nodes > target_nodes {
                upper_bound = mid_mismatch;

                let index = upper_bound
                    .matching_eigenvalues
                    .partition_point(|&x| x < 0.);
                upper_eigenvalue = upper_bound.matching_eigenvalues.get(index - 1);
            } else {
                lower_bound = mid_mismatch;

                let index = lower_bound
                    .matching_eigenvalues
                    .partition_point(|&x| x < 0.);
                lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);
            }
        }

        let result = brent_root_method(
            [lower_bound.energy.to_au(), *lower_eigenvalue.unwrap()],
            [upper_bound.energy.to_au(), *upper_eigenvalue.unwrap()],
            |x| {
                let mismatch = self.bound_mismatch(Energy(x, Au));

                let index = mismatch.matching_eigenvalues.partition_point(|&x| x < 0.);

                if mismatch.nodes > target_nodes {
                    mismatch.matching_eigenvalues[index - 1]
                } else {
                    mismatch.matching_eigenvalues[index]
                }
            },
            err,
            30,
        );

        match result {
            Ok(x) => return Energy(x, Au),
            Err(err) => panic!("{err:?}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BoundStates {
    pub energies: Vec<f64>,
    pub nodes: Vec<u64>,
    pub occupations: Option<Vec<Vec<f64>>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BoundStatesDependence<T> {
    pub parameters: Vec<T>,
    pub bound_states: Vec<BoundStates>,
}

impl BoundStates {
    // todo! fix dubious unit casting, which works only once
    pub fn with_energy_units(mut self, unit: impl EnergyUnit) -> Self {
        for energy in &mut self.energies {
            *energy = Energy(*energy, Au).to(unit).value()
        }

        self
    }
}

#[derive(Serialize)]
pub struct WaveFunctions {
    pub bounds: BoundStates,
    pub waves: Vec<WaveFunction>
}

#[derive(Serialize, Debug)]
pub struct WaveFunction {
    pub distances: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

impl WaveFunction {
    pub fn reverse(&mut self) {
        self.distances.reverse();
        self.values.reverse();
    }

    pub fn extend(&mut self, wave: WaveFunction) {
        self.distances.extend(wave.distances);
        self.values.extend(wave.values);
    }

    pub fn normalize(mut self) -> Self {
        let normalization: f64 = self.distances.windows(2)
            .zip(self.values.windows(2))
            .map(|(x, f)| unsafe {
                let f1 = f.get_unchecked(1);
                let f0 = f.get_unchecked(0);
                let f1_norm = f1.iter().fold(0., |acc, x| acc + x * x);
                let f0_norm = f0.iter().fold(0., |acc, x| acc + x * x);

                0.5 * (x.get_unchecked(1) - x.get_unchecked(0)) * (f1_norm + f0_norm)
            })
            .sum();
        
        for v in &mut self.values {
            for p in v {
                *p /= normalization.sqrt()
            }
        }

        self
    }

    pub fn occupations(&self) -> Vec<f64> {
        self.distances.windows(2)
            .zip(self.values.windows(2))
            .fold(vec![0.; self.values[0].len()], |mut acc, (d, v)| {
                for i in 0..acc.len() {
                    acc[i] += 0.5 * (d[1] - d[0]) * (v[1][i].powi(2) + v[0][i].powi(2))
                }
            
            acc
        })
    }
}