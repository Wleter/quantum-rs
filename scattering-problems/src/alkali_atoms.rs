use abm::{DoubleHifiProblemBuilder, abm_states::HifiStates};
use clebsch_gordan::hu32;
use quantum::operator_diagonal_mel;
use scattering_solver::{
    boundary::Asymptotic,
    potentials::{
        dispersion_potential::Dispersion,
        masked_potential::MaskedPotential,
        multi_diag_potential::Diagonal,
        pair_potential::PairPotential,
        potential::{MatPotential, Potential, SimplePotential},
    },
    utility::AngMomentum,
};

use crate::{FieldScatteringProblem, IndexBasisDescription, ScatteringProblem};

#[derive(Clone)]
pub struct AlkaliAtomsProblemBuilder<P, V>
where
    P: SimplePotential,
    V: SimplePotential,
{
    hifi_problem: DoubleHifiProblemBuilder,
    singlet_potential: V,
    triplet_potential: P,
}

impl<P, V> AlkaliAtomsProblemBuilder<P, V>
where
    P: SimplePotential,
    V: SimplePotential,
{
    pub fn new(
        hifi_problem: DoubleHifiProblemBuilder,
        singlet_potential: V,
        triplet_potential: P,
    ) -> Self {
        assert!(hifi_problem.first.s == hu32!(1 / 2));
        assert!(hifi_problem.second.s == hu32!(1 / 2));

        Self {
            hifi_problem,
            singlet_potential,
            triplet_potential,
        }
    }

    pub fn build(
        self,
        magnetic_field: f64,
    ) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
        let hifi = self.hifi_problem.build();
        let basis = hifi.get_basis();

        let hifi_states = hifi.states_at(magnetic_field);

        let singlet_masking = operator_diagonal_mel!(&basis, |[e: HifiStates::ElectronSpin]| {
            if e.s == 0 { 1. } else { 0. }
        });
        let singlet_masking = hifi_states.1.transpose() * singlet_masking.as_ref() * &hifi_states.1;
        let singlet_potential = MaskedPotential::new(self.singlet_potential, singlet_masking);

        let triplet_masking = operator_diagonal_mel!(&basis, |[e: HifiStates::ElectronSpin]| {
            if e.s == 1 { 1. } else { 0. }
        });
        let triplet_masking = hifi_states.1.transpose() * triplet_masking.as_ref() * &hifi_states.1;
        let triplet_potential = MaskedPotential::new(self.triplet_potential, triplet_masking);

        let hifi_potential = hifi_states
            .0
            .iter()
            .map(|e| Dispersion::new(e.to_au(), 0))
            .collect();
        let hifi_potential = Diagonal::from_vec(hifi_potential);

        let potential = PairPotential::new(triplet_potential, singlet_potential);
        let full_potential = PairPotential::new(hifi_potential, potential);

        ScatteringProblem {
            asymptotic: Asymptotic {
                centrifugal: vec![AngMomentum(0); full_potential.size()],
                entrance: 0,
                channel_energies: hifi_states.0.iter().map(|x| x.to_au()).collect(),
                channel_states: hifi_states.1,
            },
            potential: full_potential,
            basis_description: IndexBasisDescription,
        }
    }
}

// very temporary fix soon todo!
impl<P, V> FieldScatteringProblem<IndexBasisDescription> for AlkaliAtomsProblemBuilder<P, V> 
where
    P: SimplePotential + Clone,
    V: SimplePotential + Clone,
{
    fn levels(&self, field: f64, _l: Option<u32>) -> (Vec<f64>, faer::Mat<f64>) {
        let hifi = self.clone().hifi_problem.build();
        let hifi_states = hifi.states_at(field);

        let levels = hifi_states.0.iter().map(|x| x.to_au()).collect();

        (levels, hifi_states.1)
    }

    fn scattering_for(&self, field: f64) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
        self.clone().build(field)
    }
}