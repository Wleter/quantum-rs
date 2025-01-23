use scattering_solver::{boundary::Asymptotic, potentials::potential::Potential};

pub mod alkali_atoms;
pub mod alkali_rotor_atom;
pub mod utility;
pub mod rotor_atom;
pub mod potential_interpolation;
pub mod operators;
pub mod alkali_rotor;
pub mod angular_block;
pub mod uncoupled_alkali_rotor_atom;

pub struct ScatteringProblem<P: Potential, B: BasisDescription> {
    pub potential: P,
    pub asymptotic: Asymptotic,
    pub basis_description: B
}

pub trait FieldDependentScatteringProblem<P: Potential, B: BasisDescription> {
    fn scattering_for(&self, field: f64) -> ScatteringProblem<P, B>;
}

pub trait BasisDescription {
    type BasisElement;

    fn index_for(&self, channel: &Self::BasisElement) -> usize;
}
pub struct IndexBasisDescription;

impl BasisDescription for IndexBasisDescription {
    type BasisElement = usize;

    fn index_for(&self, &channel: &usize) -> usize {
        channel
    }
}
