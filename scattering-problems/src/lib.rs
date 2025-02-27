use faer::Mat;
use scattering_solver::{
    boundary::Asymptotic,
    potentials::potential::{MatPotential, Potential},
};

pub mod alkali_atoms;
pub mod alkali_rotor;
pub mod alkali_rotor_atom;
pub mod angular_block;
pub mod potential_interpolation;
pub mod rkhs_interpolation;
pub mod rotor_atom;
pub mod uncoupled_alkali_rotor_atom;
pub mod utility;

pub struct ScatteringProblem<P: Potential, B: BasisDescription> {
    pub potential: P,
    pub asymptotic: Asymptotic,
    pub basis_description: B,
}

pub trait FieldScatteringProblem<B: BasisDescription> {
    fn levels(&self, field: f64, l: Option<u32>) -> (Vec<f64>, Mat<f64>);

    fn scattering_for(&self, field: f64) -> ScatteringProblem<impl MatPotential, B>;
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
