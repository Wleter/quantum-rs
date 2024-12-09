use faer::Mat;
use quantum::{params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, state::State, state_type::StateType, States, StatesBasis}};
use scattering_solver::potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, pair_potential::PairPotential, potential::{Potential, SimplePotential}};

use crate::utility::{percival_coef, RotorJMax, RotorJTot, RotorLMax};

#[derive(Clone, Copy, PartialEq)]
pub enum RotorAtomSFStates {
    SystemL,
    RotorJ,
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
        Self {
            potential
        }
    }

    pub fn build_space_fixed(self, particles: &Particles) -> impl Potential<Space = Mat<f64>> {
        let l_max = particles.get::<RotorLMax>().expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>().expect("Did not find RotorJMax parameter in particles").0;
        let j_tot = particles.get::<RotorJTot>().map_or(0, |x| x.0);
        // todo! possibly change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>().expect("Did not find RotConst parameter in the particles").0;

        let all_even = self.potential.iter().all(|(lambda, _)| lambda & 1 == 0);

        let ls = if all_even { (0..=l_max).step_by(2).collect() } else { (0..=l_max).collect() };
        let system_l = State::new(RotorAtomSFStates::SystemL, ls);
        let js = if all_even { (0..=j_max).step_by(2).collect() } else { (0..=j_max).collect() };
        let rotor_j = State::new(RotorAtomSFStates::RotorJ, js);

        let mut rotor_states = States::default();
        rotor_states.push_state(StateType::Irreducible(system_l))
            .push_state(StateType::Irreducible(rotor_j));
        let rotor_basis: StatesBasis<RotorAtomSFStates, u32> = rotor_states.iter_elements()
            .filter(|a| {
                let j = a.values[0];
                let l = a.values[1];

                l + j >= j_tot && (l as i32 - j as i32).unsigned_abs() <= j_tot
            })
            .collect();

        let l_centrifugal_mask = Operator::from_diagonal_mel(&rotor_basis, [RotorAtomSFStates::SystemL], |[l]| {
            (l.1 * (l.1 + 1)) as f64
        }).into_backed();
        let l_potential = Dispersion::new(0.5 / particles.red_mass(), -2);

        let j_centrifugal = Operator::from_diagonal_mel(&rotor_basis, [RotorAtomSFStates::RotorJ], |[j]| {
            (j.1 * (j.1 + 1)) as f64
        }).into_backed();
        let j_centrifugal_mask = j_centrifugal;
        let j_potential = Dispersion::new(rot_const, 0);

        let mut rotor_centrifugal = Composite::new(MaskedPotential::new(l_potential, l_centrifugal_mask));
        rotor_centrifugal.add_potential(MaskedPotential::new(j_potential, j_centrifugal_mask));

        let mut potentials = self.potential.into_iter()
            .map(|(lambda, potential)| {
                let rotor_masking = Operator::from_mel(&rotor_basis, [RotorAtomSFStates::SystemL, RotorAtomSFStates::RotorJ], |[l, j]| {
                    let lj_left = (l.bra.1, j.bra.1);
                    let lj_right = (l.ket.1, j.ket.1);

                    percival_coef(lambda, lj_left, lj_right, j_tot)
                }).into_backed();

                MaskedPotential::new(potential, rotor_masking)
            });

        let mut potential = Composite::new(potentials.next().expect("No singlet potentials found"));
        for p in potentials {
            potential.add_potential(p);
        }

        PairPotential::new(rotor_centrifugal, potential)
    }
}
