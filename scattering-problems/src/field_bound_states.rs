use std::marker::PhantomData;

use faer::Mat;
use quantum::{
    params::particles::Particles,
};

use scattering_solver::{
    boundary::{Boundary, Direction}, log_derivatives::{LogDerivative, LogDerivativeReference, WaveLogDerivStorage}, numerovs::StepRule, observables::bound_states::WaveFunction, potentials::potential::Potential, propagator::{CoupledEquation, Propagator}, utility::brent_root_method
};
use serde::{Deserialize, Serialize};

use crate::{BasisDescription, FieldScatteringProblem};

// todo! refactor everything is copied from bound_states

#[derive(Clone, Debug)]
pub struct BoundFieldMismatch {
    pub nodes: u64,
    pub matching_matrix: Mat<f64>,
    pub matching_eigenvalues: Vec<f64>,
    pub field: f64,
}

pub struct FieldProblemBuilder<'a, B, F, S, Prop>
where
    B: BasisDescription,
    F: FieldScatteringProblem<B>,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
{
    particles: Option<&'a Particles>,
    field_problem: Option<&'a F>,

    step_rule: Option<S>,
    phantom: PhantomData<(Prop, B)>,

    r_range: [f64; 3],
}

impl<'a, B, F, S, Prop> Default for FieldProblemBuilder<'a, B, F, S, Prop>
where
    B: BasisDescription,
    F: FieldScatteringProblem<B>,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
{
    fn default() -> Self {
        Self {
            particles: Default::default(),
            field_problem: Default::default(),

            step_rule: Default::default(),
            phantom: PhantomData,

            r_range: Default::default(),
        }
    }
}

impl<'a, B, F, S, Prop> FieldProblemBuilder<'a, B, F, S, Prop>
where
    B: BasisDescription,
    F: FieldScatteringProblem<B>,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
{
    pub fn new(particles: &'a Particles, field_problem: &'a F) -> Self {
        let mut problem = FieldProblemBuilder::default();

        problem.particles = Some(particles);
        problem.field_problem = Some(field_problem);

        problem
    }

    pub fn with_propagation(mut self, step_rule: S, _prop: Prop) -> Self {
        self.step_rule = Some(step_rule);

        self
    }

    pub fn with_range(mut self, r_min: f64, r_match: f64, r_max: f64) -> Self {
        self.r_range = [r_min, r_match, r_max];

        self
    }

    pub fn build(self) -> FieldProblem<'a, B, F, S, Prop> {
        let particles = self
            .particles
            .expect("Did not found particles in BoundBuilder");
        let field_problem = self
            .field_problem
            .expect("Did not found potential in BoundBuilder");

        let step_rule = self
            .step_rule
            .expect("Did not found step rule in BoundBuilder");

        FieldProblem {
            particles,
            field_problem,
            step_rule: step_rule,
            phantom: self.phantom,
            r_min: self.r_range[0],
            r_match: self.r_range[1],
            r_max: self.r_range[2],
        }
    }
}

pub struct FieldProblem<'a, B, F, S, Prop>
where
    B: BasisDescription,
    F: FieldScatteringProblem<B>,
    S: StepRule<Mat<f64>>,
    Prop: LogDerivativeReference,
{
    particles: &'a Particles,
    field_problem: &'a F,

    step_rule: S,
    phantom: PhantomData<(Prop, B)>,

    r_min: f64,
    r_match: f64,
    r_max: f64,
}

impl<'a, B, F, S, Prop> FieldProblem<'a, B, F, S, Prop>
where
    B: BasisDescription,
    F: FieldScatteringProblem<B>,
    S: StepRule<Mat<f64>> + Clone,
    Prop: LogDerivativeReference,
{
    pub fn bound_mismatch(&self, field: f64) -> BoundFieldMismatch {
        let problem = self.field_problem.scattering_for(field);
        let mut particles = self.particles.clone();
        particles.insert(problem.asymptotic);

        let eq = CoupledEquation::from_particles(&problem.potential, &particles);
        let boundary = Boundary::new_exponential_vanishing(self.r_max, &eq);

        let mut propagator =
            LogDerivative::<_, Prop>::new(eq.clone(), boundary, self.step_rule.clone());
        let solution_in = propagator.propagate_to(self.r_match);

        let boundary =
            Boundary::new_multi_vanishing(self.r_min, Direction::Outwards, problem.potential.size());
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

        BoundFieldMismatch {
            nodes,
            matching_matrix,
            matching_eigenvalues: eigenvalues,
            field,
        }
    }

    pub fn bound_states(
        &self,
        field_range: (f64, f64),
        err: f64,
    ) -> FieldBoundStates {
        let lower_mismatch = self.bound_mismatch(field_range.0);
        let upper_mismatch = self.bound_mismatch(field_range.1);

        let mut bound_fields = vec![];
        let mut bound_nodes = vec![];

        let lower_node = lower_mismatch.nodes;
        let upper_node = upper_mismatch.nodes;
        let states_no = (upper_node as i64 - lower_node as i64).abs() as usize;

        if states_no == 0 {
            return FieldBoundStates {
                fields: vec![],
                nodes: vec![],
            }
        }

        let mut mismatch_node = vec![None; states_no + 1];
        mismatch_node[0] = Some(lower_mismatch);
        mismatch_node[states_no] = Some(upper_mismatch);

        let mut target_nodes = upper_node - 1;
        while target_nodes >= lower_node {
            let bound_field = self.search_state(lower_node, target_nodes, &mut mismatch_node, err);

            bound_fields.push(bound_field);
            bound_nodes.push(target_nodes);

            target_nodes -= 1;
        }

        FieldBoundStates {
            fields: bound_fields,
            nodes: bound_nodes,
        }
    }

    pub fn bound_waves(&self, bounds: &FieldBoundStates) -> impl Iterator<Item = WaveFunction> {
        bounds.fields.iter()
            .map(move |&f| {
                let mut particles = self.particles.clone();
                let problem = self.field_problem.scattering_for(f);
                particles.insert(problem.asymptotic);

                let eq = CoupledEquation::from_particles(&problem.potential, &particles);
                let boundary = Boundary::new_exponential_vanishing(self.r_max, &eq);

                let mut propagator_in = LogDerivative::<_, Prop>::new(eq.clone(), boundary, self.step_rule.clone());
                propagator_in.with_storage(WaveLogDerivStorage::new(true));
                let solution_in = propagator_in.propagate_to(self.r_match);

                let boundary = Boundary::new_multi_vanishing(self.r_min, Direction::Outwards, problem.potential.size());
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

                wave_out
            })
    }

    fn search_state(
        &self,
        index_offset: u64,
        target_nodes: u64,
        mismatch_node: &mut Vec<Option<BoundFieldMismatch>>,
        err: f64,
    ) -> f64 {
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
            let field_mid = (upper_bound.field + lower_bound.field) / 2.;
            if upper_bound.field - lower_bound.field < err {
                return field_mid;
            }

            let mid_mismatch = self.bound_mismatch(field_mid);

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
            [lower_bound.field, *lower_eigenvalue.unwrap()],
            [upper_bound.field, *upper_eigenvalue.unwrap()],
            |x| {
                let mismatch = self.bound_mismatch(x);

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
            Ok(x) => return x,
            Err(err) => panic!("{err:?}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FieldBoundStates {
    pub fields: Vec<f64>,
    pub nodes: Vec<u64>,
}

#[derive(Serialize)]
pub struct FieldWaveFunctions {
    pub bounds: FieldBoundStates,
    pub waves: Vec<WaveFunction>
}
