use abm::{abm_states::{ABMStates, HifiStates}, ABMHifiProblem, ABMProblemBuilder, ABMVibrational, DoubleHifiProblemBuilder, HifiProblemBuilder, Symmetry};
use faer::Mat;
use pyo3::prelude::*;
use quantum::units::{energy_units::Energy, Au};

pub const BOHR_MAG: f64 = 0.5 / 2.350517567e9;
pub const NUCLEAR_MAG: f64 = 0.5 / 1836.0 / 2.350517567e9;

#[pyclass(name = "HifiProblemBuilder")]
struct HifiProblemBuilderPy(HifiProblemBuilder);

#[pymethods]
impl HifiProblemBuilderPy {
    #[new]
    pub fn new(s: u32, i: u32) -> Self {
        Self(HifiProblemBuilder::new(s, i))
    }

    pub fn with_projection(&mut self, double_projection: i32) {
        let hifi = self.0.clone();

        self.0 = hifi.with_total_projection(double_projection);
    }

    pub fn with_custom_bohr_magneton(&mut self, gamma_e: f64) {
        let hifi = self.0.clone();

        self.0 = hifi.with_custom_bohr_magneton(gamma_e);
    }

    pub fn with_nuclear_magneton(&mut self, gamma_i: f64) {
        let hifi = self.0.clone();

        self.0 = hifi.with_nuclear_magneton(gamma_i);
    }

    pub fn with_hyperfine_coupling(&mut self, a_hifi: f64) {
        let hifi = self.0.clone();

        self.0 = hifi.with_hyperfine_coupling(a_hifi);
    }

    pub fn build(&mut self) -> HifiProblemPy {
        HifiProblemPy(self.0.clone().build())
    }
}

#[pyclass(name = "DoubleHifiProblemBuilder")]
struct DoubleHifiProblemBuilderPy(DoubleHifiProblemBuilder);

#[pymethods]
impl DoubleHifiProblemBuilderPy {
    #[new]
    pub fn new(first: PyRef<HifiProblemBuilderPy>, second: PyRef<HifiProblemBuilderPy>) -> Self {
        Self(DoubleHifiProblemBuilder::new(first.0.clone(), second.0.clone()))
    }

    #[staticmethod]
    #[pyo3(signature = (single, symmetry = "none"))]
    pub fn new_homo(single: PyRef<HifiProblemBuilderPy>, symmetry: &str) -> Self {
        let symmetry = match symmetry {
            "none" => Symmetry::None,
            "fermionic" => Symmetry::Fermionic,
            "bosonic" => Symmetry::Bosonic,
            _ => panic!("only none/fermionic/bosonic values are allowed for symmetry type")
        };

        Self(DoubleHifiProblemBuilder::new_homo(single.0.clone(), symmetry))
    }

    pub fn with_projection(&mut self, double_projection: i32) {
        let hifi = self.0.clone();

        self.0 = hifi.with_projection(double_projection);
    }

    pub fn build(&mut self) -> HifiProblemPy {
        HifiProblemPy(self.0.clone().build())
    }
}

#[pyclass(name = "ABMProblemBuilder")]
struct ABMProblemBuilderPy(ABMProblemBuilder);

#[pymethods]
impl ABMProblemBuilderPy {
    #[new]
    pub fn new(first: PyRef<HifiProblemBuilderPy>, second: PyRef<HifiProblemBuilderPy>) -> Self {
        Self(ABMProblemBuilder::new(first.0.clone(), second.0.clone()))
    }

    #[staticmethod]
    #[pyo3(signature = (single, symmetry = "none"))]
    pub fn new_homo(single: PyRef<HifiProblemBuilderPy>, symmetry: &str) -> Self {
        let symmetry = match symmetry {
            "none" => Symmetry::None,
            "fermionic" => Symmetry::Fermionic,
            "bosonic" => Symmetry::Bosonic,
            _ => panic!("only none/fermionic/bosonic values are allowed for symmetry type")
        };

        Self(ABMProblemBuilder::new_homo(single.0.clone(), symmetry))
    }

    pub fn with_projection(&mut self, double_projection: i32) {
        let abm = self.0.clone();

        self.0 = abm.with_projection(double_projection);
    }

    pub fn with_vibrational(&mut self, singlet_states: Vec<f64>, triplet_states: Vec<f64>, fc_factors: Vec<f64>) {
        let abm = self.0.clone();

        assert!(singlet_states.len().pow(2) == fc_factors.len(), "non-compatible franck-condon factor matrix size");

        let singlet_states: Vec<Energy<Au>> = singlet_states.iter().map(|&x| Energy(x, Au)).collect();
        let triplet_states: Vec<Energy<Au>> = triplet_states.iter().map(|&x| Energy(x, Au)).collect();
        let n = singlet_states.len();
        let fc_factors = Mat::from_fn(n, n, |row, col| fc_factors[row * n + col]);

        let vibrational = ABMVibrational::new(singlet_states, triplet_states, fc_factors);

        self.0 = abm.with_vibrational(vibrational);
    }

    pub fn build(&mut self) -> ABMProblemPy {
        ABMProblemPy(self.0.clone().build())
    }
}

#[pyclass(name = "HifiProblem")]
pub struct HifiProblemPy(ABMHifiProblem<HifiStates, i32>);

#[pymethods]
impl HifiProblemPy {
    pub fn states_at(&mut self, magnetic_field: f64) -> Vec<f64> {
        let (values, _) = self.0.states_at(magnetic_field);

        values.into_iter().map(|x| x.value()).collect()
    }

    pub fn states_range(&mut self, magnetic_fields: Vec<f64>) -> Vec<Vec<f64>> {
        magnetic_fields.iter().map(|&magnetic_field| {
            let (values, _) = self.0.states_at(magnetic_field);
    
            values.into_iter().map(|x| x.value()).collect()
        })
        .collect()
    }
}

#[pyclass(name = "ABMProblem")]
pub struct ABMProblemPy(ABMHifiProblem<ABMStates, i32>);

#[pymethods]
impl ABMProblemPy {
    pub fn states_at(&mut self, magnetic_field: f64) -> Vec<f64> {
        let (values, _) = self.0.states_at(magnetic_field);

        values.into_iter().map(|x| x.value()).collect()
    }

    pub fn states_range(&mut self, magnetic_fields: Vec<f64>) -> Vec<Vec<f64>> {
        magnetic_fields.iter().map(|&magnetic_field| {
            let (values, _) = self.0.states_at(magnetic_field);
    
            values.into_iter().map(|x| x.value()).collect()
        })
        .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn abm_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HifiProblemBuilderPy>()?;
    m.add_class::<DoubleHifiProblemBuilderPy>()?;
    m.add_class::<ABMProblemBuilderPy>()?;

    m.add("BOHR_MAG", BOHR_MAG)?;
    m.add("NUCLEAR_MAG", NUCLEAR_MAG)?;

    m.add_class::<HifiProblemPy>()?;
    m.add_class::<ABMProblemPy>()?;

    Ok(())
}
