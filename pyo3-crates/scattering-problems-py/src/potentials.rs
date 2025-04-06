use pyo3::prelude::*;
use scattering_solver::potentials::{dispersion_potential::Dispersion, potential::Potential};

#[pyclass]
#[pyo3(name = "Potential")]
pub struct PotentialPy(pub Box<dyn Potential<Space = f64> + Send + Sync>);

#[pymethods]
impl PotentialPy {
    #[staticmethod]
    pub fn dispersion(cn: f64, n: i32) -> Self {
        PotentialPy(Box::new(Dispersion::new(cn, n)))
    }
}
