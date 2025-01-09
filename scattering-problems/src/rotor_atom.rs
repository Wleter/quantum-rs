use clebsch_gordan::half_integer::HalfU32;
use faer::Mat;
use quantum::{params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, state::State, state_type::StateType, States, StatesBasis}};
use scattering_solver::{boundary::Asymptotic, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, pair_potential::PairPotential, potential::{MatPotential, SimplePotential}}, utility::AngMomentum};

use crate::{utility::{percival_coef, RotorJMax, RotorJTot, RotorLMax}, BasisDescription, ScatteringProblem};

#[derive(Clone, Copy, PartialEq)]
pub enum RotorAtomStates {
    SystemL,
    RotorJ,
}

#[derive(Clone)]
pub struct RotorAtomProblemBuilder<P>
where 
    P: SimplePotential,
{
    potential: Vec<(u32, P)>
}

impl<P> RotorAtomProblemBuilder<P> 
where 
    P: SimplePotential,
{
    pub fn new(potential: Vec<(u32, P)>) -> Self {
        Self {
            potential
        }
    }

    pub fn build(self, particles: &Particles) -> ScatteringProblem<impl MatPotential, RotorAtomBasisDescription> {
        // todo! possibly change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>()
            .expect("Did not find RotConst parameter in the particles").0;
        let j_tot = particles.get::<RotorJTot>().map_or(0, |x| x.0);

        let rotor_basis = self.basis(particles);

        let j_centrifugal_mask = Operator::from_diagonal_mel(&rotor_basis, [RotorAtomStates::RotorJ], |[j]| {
            (j.1 * (j.1 + 1)) as f64
        }).into_backed();
        let j_potential = Dispersion::new(rot_const, 0);

        let rotor_centrifugal = MaskedPotential::new(j_potential, j_centrifugal_mask);

        let mut potentials = self.potential.into_iter()
            .map(|(lambda, potential)| {
                let rotor_masking = Operator::from_mel(&rotor_basis, [RotorAtomStates::SystemL, RotorAtomStates::RotorJ], |[l, j]| {
                    let lj_left = (HalfU32::from_doubled(2 * l.bra.1), HalfU32::from_doubled(2 * j.bra.1));
                    let lj_right = (HalfU32::from_doubled(2 * l.ket.1), HalfU32::from_doubled(2 * j.ket.1));

                    percival_coef(lambda, lj_left, lj_right, HalfU32::from_doubled(2 * j_tot))
                }).into_backed();

                MaskedPotential::new(potential, rotor_masking)
            });

        let mut potential = Composite::new(potentials.next().expect("No singlet potentials found"));
        for p in potentials {
            potential.add_potential(p);
        }

        let full_potential = PairPotential::new(rotor_centrifugal, potential);

        let angular_momenta = rotor_basis.iter()
            .map(|x| AngMomentum(x.values[0]))
            .collect();
        
        let channel_energies = rotor_basis.iter()
            .map(|x| rot_const * (x.values[1] * (x.values[1] + 1)) as f64)
            .collect();

        let asymptotic = Asymptotic {
            centrifugal: angular_momenta,
            entrance: 0,
            channel_energies,
            channel_states: Mat::identity(rotor_basis.len(), rotor_basis.len()),
        };

        let description = RotorAtomBasisDescription {
            basis: rotor_basis,
            j_tot
        };

        ScatteringProblem {
            potential: full_potential,
            asymptotic,
            basis_description: description
        }
    }

    fn basis(&self, particles: &Particles) -> StatesBasis<RotorAtomStates, u32> {
        let l_max = particles.get::<RotorLMax>()
            .expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>()
            .expect("Did not find RotorJMax parameter in particles").0;
        let j_tot = particles.get::<RotorJTot>().map_or(0, |x| x.0);

        let all_even = self.potential.iter().all(|(lambda, _)| lambda & 1 == 0);

        let ls = (0..=l_max)
            .step_by(if all_even { 2 } else { 1 } )
            .collect();
        let system_l = State::new(RotorAtomStates::SystemL, ls);

        let js = (0..=j_max)
            .step_by(if all_even { 2 } else { 1 } )
            .collect();
        let rotor_j = State::new(RotorAtomStates::RotorJ, js);

        let mut rotor_states = States::default();
        rotor_states.push_state(StateType::Irreducible(system_l))
            .push_state(StateType::Irreducible(rotor_j));
        
        rotor_states.iter_elements()
            .filter(|a| {
                let j = a.values[0];
                let l = a.values[1];

                l + j >= j_tot && (l as i32 - j as i32).unsigned_abs() <= j_tot
            })
            .collect()
    }
}

pub struct RotorAtomBasisDescription {
    pub basis: StatesBasis<RotorAtomStates, u32>,
    pub j_tot: u32
}

pub struct RotorAtomBasisElement {
    pub l: u32,
    pub j: u32
}

impl BasisDescription for RotorAtomBasisDescription {
    type BasisElement = RotorAtomBasisElement;
    
    fn index_for(&self, channel: &Self::BasisElement) -> usize {
        self.basis.iter()
            .enumerate()
            .find(|(_, s)| s.values[0] == channel.l && s.values[1] == channel.j)
            .expect("Did not found entrance channel")
            .0
    }
}