use abm::{abm_states::HifiStates, DoubleHifiProblemBuilder};
use clebsch_gordan::hu32;
use quantum::{cast_variant, states::operator::Operator};
use scattering_solver::{boundary::Asymptotic, potentials::{dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::{MatPotential, Potential, SimplePotential}}, utility::AngMomentum};

use crate::{IndexBasisDescription, ScatteringProblem};

#[derive(Clone)]
pub struct AlkaliAtomsProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    hifi_problem: DoubleHifiProblemBuilder,
    triplet_potential: P,
    singlet_potential: V,
}

impl<P, V> AlkaliAtomsProblemBuilder<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub fn new(hifi_problem: DoubleHifiProblemBuilder, triplet_potential: P, singlet_potential: V) -> Self {
        assert!(hifi_problem.first.s == hu32!(1/2));
        assert!(hifi_problem.second.s == hu32!(1/2));

        Self {
            hifi_problem,
            triplet_potential,
            singlet_potential
        }
    }

    pub fn build(self, magnetic_field: f64) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
        let hifi = self.hifi_problem.build();
        let basis = hifi.get_basis();
        
        let hifi_states = hifi.states_at(magnetic_field);

        let triplet_masking = Operator::from_diagonal_mel(basis, [HifiStates::ElectronSpin(hu32!(0))], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronSpin);

            if spin_e == hu32!(1) { 1. } else { 0. }
        });
        let triplet_masking = hifi_states.1.transpose() * triplet_masking.as_ref() * &hifi_states.1;
        let triplet_potential = MaskedPotential::new(self.triplet_potential, triplet_masking);

        let singlet_masking = Operator::from_diagonal_mel(basis, [HifiStates::ElectronSpin(hu32!(0))], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronSpin);

            if spin_e == hu32!(0) { 1. } else { 0. }
        });
        let singlet_masking = hifi_states.1.transpose() * singlet_masking.as_ref() * &hifi_states.1;
        let singlet_potential = MaskedPotential::new(self.singlet_potential, singlet_masking);

        let hifi_potential = hifi_states.0.iter()
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
            basis_description: IndexBasisDescription
        }
    }
}
