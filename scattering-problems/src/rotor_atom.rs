use faer::Mat;
use quantum::{
    cast_variant, operator_diagonal_mel, operator_mel,
    params::{Params, particle_factory::RotConst},
    states::{
        States, StatesBasis,
        braket::Braket,
        state::{StateBasis, into_variant},
    },
};
use scattering_solver::{
    boundary::Asymptotic,
    potentials::{
        composite_potential::Composite,
        dispersion_potential::Dispersion,
        masked_potential::MaskedPotential,
        pair_potential::PairPotential,
        potential::{MatPotential, SimplePotential, SubPotential},
    },
    utility::AngMomentum,
};

use crate::{
    BasisDescription, ScatteringProblem,
    utility::{AngularPair, percival_coef},
};

use RotorAtomStates::*;

#[derive(Clone, Copy, PartialEq)]
pub enum RotorAtomStates {
    SystemL(u32),
    RotorN(u32),
}

#[derive(Clone, Debug, Copy, Default)]
pub struct RotorAtomBasisRecipe {
    pub l_max: u32,
    pub n_max: u32,
    pub n_tot: u32,
}

#[derive(Clone)]
pub struct RotorAtomProblemBuilder<P>
where
    P: SimplePotential,
{
    potential: Vec<(u32, P)>,
}

impl<P> RotorAtomProblemBuilder<P>
where
    P: SimplePotential,
{
    pub fn new(potential: Vec<(u32, P)>) -> Self {
        Self { potential }
    }

    pub fn build(
        self,
        params: &Params,
        basis_recipe: &RotorAtomBasisRecipe,
    ) -> ScatteringProblem<impl MatPotential + SubPotential + use<P>, RotorAtomBasisDescription> {
        let rot_const = params
            .get::<RotConst>()
            .expect("Did not find RotConst parameter in the params")
            .0;
        let n_tot = basis_recipe.n_tot;

        let rotor_basis = self.basis(basis_recipe);

        let n_centrifugal_mask = operator_diagonal_mel!(&rotor_basis, |[n: RotorN]|
            (n * (n + 1)) as f64
        )
        .into_backed();

        let n_potential = Dispersion::new(rot_const, 0);
        let rotor_centrifugal = MaskedPotential::new(n_potential, n_centrifugal_mask);

        let potentials = self
            .potential
            .into_iter()
            .map(|(lambda, potential)| {
                let rotor_masking = operator_mel!(&rotor_basis,
                    |[l: SystemL, n: RotorN]| {
                        let ang = Braket {
                            bra: AngularPair {
                                l: l.bra.into(),
                                n: n.bra.into()
                            },
                            ket: AngularPair {
                                l: l.ket.into(),
                                n: n.ket.into()
                            }
                        };

                        percival_coef(lambda, ang, n_tot.into())
                    }
                )
                .into_backed();

                MaskedPotential::new(potential, rotor_masking)
            })
            .collect();

        let potential = Composite::from_vec(potentials);

        let full_potential = PairPotential::new(rotor_centrifugal, potential);

        let angular_momenta = rotor_basis
            .iter()
            .map(|x| {
                let l = cast_variant!(x[0], SystemL);
                AngMomentum(l)
            })
            .collect();

        let channel_energies = rotor_basis
            .iter()
            .map(|x| {
                let n = cast_variant!(x[1], RotorN);
                rot_const * (n * (n + 1)) as f64
            })
            .collect();

        let asymptotic = Asymptotic {
            centrifugal: angular_momenta,
            entrance: 0,
            channel_energies,
            channel_states: Mat::identity(rotor_basis.len(), rotor_basis.len()),
        };

        let description = RotorAtomBasisDescription {
            basis: rotor_basis,
            n_tot,
        };

        ScatteringProblem {
            potential: full_potential,
            asymptotic,
            basis_description: description,
        }
    }

    fn basis(&self, basis_recipe: &RotorAtomBasisRecipe) -> StatesBasis<RotorAtomStates> {
        let l_max = basis_recipe.l_max;
        let n_max = basis_recipe.n_max;
        let n_tot = basis_recipe.n_tot;

        let all_even = self.potential.iter().all(|(lambda, _)| lambda & 1 == 0);

        let ls: Vec<u32> = (0..=l_max).step_by(if all_even { 2 } else { 1 }).collect();
        let system_l = into_variant(ls, SystemL);

        let ns = (0..=n_max).step_by(if all_even { 2 } else { 1 }).collect();
        let rotor_n = into_variant(ns, RotorN);

        let mut rotor_states = States::default();
        rotor_states
            .push_state(StateBasis::new(system_l))
            .push_state(StateBasis::new(rotor_n));

        rotor_states
            .iter_elements()
            .filter(|b| {
                let l = cast_variant!(b[0], SystemL);
                let n = cast_variant!(b[1], RotorN);

                l + n >= n_tot && (l as i32 - n as i32).unsigned_abs() <= n_tot
            })
            .collect()
    }
}

pub struct RotorAtomBasisDescription {
    pub basis: StatesBasis<RotorAtomStates>,
    pub n_tot: u32,
}

impl BasisDescription for RotorAtomBasisDescription {
    type BasisElement = AngularPair;

    fn index_for(&self, channel: &Self::BasisElement) -> usize {
        self.basis
            .iter()
            .enumerate()
            .find(|(_, b)| {
                let l = cast_variant!(b[0], SystemL);
                let n = cast_variant!(b[1], RotorN);

                channel.l == l && channel.n == n
            })
            .expect("Did not found entrance channel")
            .0
    }
}
