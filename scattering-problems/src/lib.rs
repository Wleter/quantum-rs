use faer::Mat;
use scattering_solver::{boundary::Asymptotic, potentials::potential::Potential};

pub mod alkali_atoms;
pub mod alkali_rotor_atom;
pub mod utility;
pub mod rotor_atom;
pub mod potential_interpolation;

pub struct ScatteringProblem<P: Potential<Space = Mat<f64>>> {
    pub potential: P,
    pub asymptotic: Asymptotic,
}