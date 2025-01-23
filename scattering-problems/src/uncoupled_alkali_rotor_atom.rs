use abm::{get_hifi, get_zeeman_prop, utility::diagonalize};
use clebsch_gordan::{clebsch_gordan, half_i32, half_integer::{HalfI32, HalfU32}, half_u32, wigner_3j};
use faer::Mat;
use quantum::{cast_variant, params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, spins::{spin_projections, Spin, SpinOperators}, state::State, state_type::StateType, States, StatesBasis}};
use scattering_solver::{boundary::Asymptotic, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::{MatPotential, SimplePotential}}, utility::AngMomentum};

use crate::{alkali_rotor_atom::AlkaliRotorAtomProblemBuilder, angular_block::{AngularBlock, AngularBlocks}, cast_spin_braket, utility::{AnisoHifi, GammaSpinRot, RotorJMax, RotorLMax}, BasisDescription, IndexBasisDescription, ScatteringProblem};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UncoupledSpinRotorAtom {
    SystemL(HalfU32),
    RotorN(HalfU32),
    RotorS(HalfU32),
    RotorI(HalfU32),
    AtomS(HalfU32),
    AtomI(HalfU32)
}

impl<P, V> AlkaliRotorAtomProblemBuilder<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub fn build_uncoupled(self, particles: &Particles) -> UncoupledAlkaliRotorAtomProblem<P, V> {
        // todo! change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>()
            .expect("Did not find RotConst parameter in the particles").0;
        let gamma_spin_rot = particles.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = particles.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;

        let l_max = particles.get::<RotorLMax>()
            .expect("Did not find SystemLMax parameter in particles").0;
        let ordered_basis = self.basis_uncoupled(particles);

        assert!(ordered_basis.is_sorted_by_key(|s| cast_variant!(s.variants[0], UncoupledSpinRotorAtom::SystemL)));

        let angular_block_basis = (0..=l_max).map(|l| {
                let l_block  = HalfU32::from_doubled(2 * l);
                let red_basis = ordered_basis.iter().filter(|s| {
                        matches!(s.variants[0], UncoupledSpinRotorAtom::SystemL(l_current) if l_block == l_current)
                    })
                    .cloned()
                    .collect::<StatesBasis<UncoupledSpinRotorAtom, HalfI32>>();

                (l, red_basis)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect::<Vec<(u32, StatesBasis<UncoupledSpinRotorAtom, HalfI32>)>>();

        let angular_blocks = angular_block_basis.iter()
            .map(|(l, basis)| {
                let j_centrifugal = Operator::from_diagonal_mel(&basis, [UncoupledSpinRotorAtom::RotorN(half_u32!(0))], |[ang]| {
                    let j = cast_variant!(ang.0, UncoupledSpinRotorAtom::RotorN).value();
        
                    rot_const * (j * (j + 1.))
                });
        
                let mut hifi = Mat::zeros(basis.len(), basis.len());
                if let Some(a_hifi) = self.hifi_problem.first.a_hifi {
                    hifi += get_hifi!(basis, UncoupledSpinRotorAtom::RotorS, UncoupledSpinRotorAtom::RotorI, a_hifi).as_ref()
                }
                if let Some(a_hifi) = self.hifi_problem.second.a_hifi {
                    hifi += get_hifi!(basis, UncoupledSpinRotorAtom::AtomS, UncoupledSpinRotorAtom::AtomI, a_hifi).as_ref()
                }
        
                let mut zeeman_prop = get_zeeman_prop!(basis, UncoupledSpinRotorAtom::RotorS, self.hifi_problem.first.gamma_e).as_ref()
                    + get_zeeman_prop!(basis, UncoupledSpinRotorAtom::AtomS, self.hifi_problem.second.gamma_e).as_ref();
                if let Some(gamma_i) = self.hifi_problem.first.gamma_i {
                    zeeman_prop += get_zeeman_prop!(basis, UncoupledSpinRotorAtom::RotorI, gamma_i).as_ref()
                }
                if let Some(gamma_i) = self.hifi_problem.second.gamma_i {
                    zeeman_prop += get_zeeman_prop!(basis, UncoupledSpinRotorAtom::AtomI, gamma_i).as_ref()
                }

                let spin_rot = Operator::from_mel(&basis, 
                    [UncoupledSpinRotorAtom::RotorN(half_u32!(0)), UncoupledSpinRotorAtom::RotorS(half_u32!(0))], 
                    |[n, s]| {
                        let n_braket = cast_spin_braket!(n, UncoupledSpinRotorAtom::RotorN);
                        let s_braket = cast_spin_braket!(s, UncoupledSpinRotorAtom::RotorS);
        
                        gamma_spin_rot * SpinOperators::dot(n_braket, s_braket)
                    }
                );

                let aniso_hifi = Operator::from_mel(&basis, 
                    [
                        UncoupledSpinRotorAtom::RotorN(half_u32!(0)), 
                        UncoupledSpinRotorAtom::RotorS(half_u32!(0)), 
                        UncoupledSpinRotorAtom::RotorI(half_u32!(0))
                    ], 
                    |[n, s, i]| {
                        let n_braket = cast_spin_braket!(n, UncoupledSpinRotorAtom::RotorN);
                        let s_braket = cast_spin_braket!(s, UncoupledSpinRotorAtom::RotorS);
                        let i_braket = cast_spin_braket!(i, UncoupledSpinRotorAtom::RotorI);
        
                        let factor = aniso_hifi / f64::sqrt(30.) * p3_factor(&s_braket.0) * p3_factor(&i_braket.0) 
                            * ((2. * n_braket.0.s.value() + 1.) * (2. * n_braket.1.s.value() + 1.)).sqrt();

                        let sign = (-1.0f64).powf(s_braket.0.s.value() - s_braket.0.ms.value() + i_braket.0.s.value() - i_braket.0.ms.value());

                        let wigners = wigner_3j(n_braket.0.s, half_u32!(2), n_braket.1.s, -n_braket.0.ms, n_braket.0.ms - n_braket.1.ms, n_braket.1.ms)
                            * wigner_3j(n_braket.0.s, half_u32!(2), n_braket.1.s, half_i32!(0), half_i32!(0), half_i32!(0))
                            * wigner_3j(half_u32!(1), half_u32!(1), half_u32!(2), s_braket.0.ms - s_braket.1.ms, i_braket.0.ms - i_braket.1.ms, n_braket.0.ms - n_braket.1.ms)
                            * wigner_3j(s_braket.0.s, half_u32!(1), s_braket.1.s, -s_braket.0.ms, s_braket.0.ms - s_braket.1.ms, s_braket.1.ms)
                            * wigner_3j(i_braket.0.s, half_u32!(1), i_braket.1.s, -i_braket.0.ms, i_braket.0.ms - i_braket.1.ms, i_braket.1.ms);

                        factor * sign * wigners
                    }
                );

                AngularBlock {
                    ang_momentum: AngMomentum(*l),
                    field_inv: hifi + aniso_hifi.as_ref() + j_centrifugal.as_ref() + spin_rot.as_ref(),
                    field_prop: zeeman_prop,
                }
            })
            .collect();

        let singlet_potentials = self.singlet_potential.into_iter()
            .map(|(lambda, pot)| {
                let lambda_h32 = HalfU32::from_doubled(2 * lambda);

                let masking_singlet = Operator::from_mel(&ordered_basis, 
                    [
                        UncoupledSpinRotorAtom::SystemL(half_u32!(0)),
                        UncoupledSpinRotorAtom::RotorN(half_u32!(0)),
                        UncoupledSpinRotorAtom::RotorS(half_u32!(0)),
                        UncoupledSpinRotorAtom::AtomS(half_u32!(0))
                    ],
                    |[l, n, s, s_a]| {
                        if s.bra.1 == s.ket.1 && s_a.bra.1 == s_a.ket.1 {
                            let l_braket = cast_spin_braket!(l, UncoupledSpinRotorAtom::SystemL);
                            let n_braket = cast_spin_braket!(n, UncoupledSpinRotorAtom::RotorN);
                            let s_braket = cast_spin_braket!(s, UncoupledSpinRotorAtom::RotorS);
                            let s_a_braket = cast_spin_braket!(s_a, UncoupledSpinRotorAtom::AtomS);
            
                            let factor = ((2. * l_braket.0.s.value() + 1.) * (2. * l_braket.1.s.value() + 1.)
                                * (2. * n_braket.0.s.value() + 1.) * (2. * n_braket.1.s.value() + 1.)).sqrt();
    
                            let wigners = wigner_3j(l_braket.0.s, lambda_h32, l_braket.1.s, half_i32!(0), half_i32!(0), half_i32!(0))
                                * wigner_3j(n_braket.0.s, lambda_h32, n_braket.1.s, half_i32!(0), half_i32!(0), half_i32!(0))
                                * clebsch_gordan(s_braket.0.s, s_braket.0.ms, s_a_braket.0.s, s_a_braket.0.ms, half_u32!(0), half_i32!(0))
                                * clebsch_gordan(s_braket.1.s, s_braket.1.ms, s_a_braket.1.s, s_a_braket.1.ms, half_u32!(0), half_i32!(0));

                            let summed = (0..=lambda)
                                .map(|m_lambda| {
                                    let m_lambda_h32 = HalfI32::from_doubled(2 * m_lambda as i32);

                                    (-1.0f64).powf(m_lambda as f64 - l_braket.0.ms.value() - n_braket.0.ms.value())
                                        * wigner_3j(l_braket.0.s, lambda_h32, l_braket.1.s, -l_braket.0.ms, -m_lambda_h32, l_braket.1.ms)
                                        * wigner_3j(n_braket.0.s, lambda_h32, n_braket.1.s, -n_braket.0.ms, m_lambda_h32, n_braket.1.ms)
                                })
                                .sum::<f64>();
    
                            factor * wigners * summed
                        } else {
                            0.
                        }
                    }
                );

                (pot, masking_singlet.into_backed())
            })
            .collect();

        let triplet_potentials = self.triplet_potential.into_iter()
            .map(|(lambda, pot)| {
                let lambda_h32 = HalfU32::from_doubled(2 * lambda);

                let masking_triplet = Operator::from_mel(&ordered_basis, 
                    [
                        UncoupledSpinRotorAtom::SystemL(half_u32!(0)),
                        UncoupledSpinRotorAtom::RotorN(half_u32!(0)),
                        UncoupledSpinRotorAtom::RotorS(half_u32!(0)),
                        UncoupledSpinRotorAtom::AtomS(half_u32!(0))
                    ],
                    |[l, n, s, s_a]| {
                        if s.bra.1 == s.ket.1 && s_a.bra.1 == s_a.ket.1 {
                            let l_braket = cast_spin_braket!(l, UncoupledSpinRotorAtom::SystemL);
                            let n_braket = cast_spin_braket!(n, UncoupledSpinRotorAtom::RotorN);
                            let s_braket = cast_spin_braket!(s, UncoupledSpinRotorAtom::RotorS);
                            let s_a_braket = cast_spin_braket!(s_a, UncoupledSpinRotorAtom::AtomS);
            
                            let factor = ((2. * l_braket.0.s.value() + 1.) * (2. * l_braket.1.s.value() + 1.)
                                * (2. * n_braket.0.s.value() + 1.) * (2. * n_braket.1.s.value() + 1.)).sqrt();
    
                            let wigners = wigner_3j(l_braket.0.s, lambda_h32, l_braket.1.s, half_i32!(0), half_i32!(0), half_i32!(0))
                                * wigner_3j(n_braket.0.s, lambda_h32, n_braket.1.s, half_i32!(0), half_i32!(0), half_i32!(0));
                            
                            let triplet_factor = (-1..=1)
                                .map(|ms_tot| {
                                    let ms_tot = HalfI32::from_doubled(2 * ms_tot);

                                    clebsch_gordan(s_braket.0.s, s_braket.0.ms, s_a_braket.0.s, s_a_braket.0.ms, half_u32!(1), ms_tot)
                                        * clebsch_gordan(s_braket.1.s, s_braket.1.ms, s_a_braket.1.s, s_a_braket.1.ms, half_u32!(1), ms_tot)
                                })
                                .sum::<f64>();

                            let summed = (0..=lambda)
                                .map(|m_lambda| {
                                    let m_lambda_h32 = HalfI32::from_doubled(2 * m_lambda as i32);

                                    (-1.0f64).powf(m_lambda as f64 - l_braket.0.ms.value() - n_braket.0.ms.value())
                                        * wigner_3j(l_braket.0.s, lambda_h32, l_braket.1.s, -l_braket.0.ms, -m_lambda_h32, l_braket.1.ms)
                                        * wigner_3j(n_braket.0.s, lambda_h32, n_braket.1.s, -n_braket.0.ms, m_lambda_h32, n_braket.1.ms)
                                })
                                .sum::<f64>();
    
                            factor * wigners * triplet_factor * summed
                        } else {
                            0.
                        }
                    }
                );

                (pot, masking_triplet.into_backed())
            })
            .collect();

        UncoupledAlkaliRotorAtomProblem {
            angular_blocks: AngularBlocks(angular_blocks),
            basis: ordered_basis,
            triplets: triplet_potentials,
            singlets: singlet_potentials,
        }
    }

    fn basis_uncoupled(&self, particles: &Particles) -> StatesBasis<UncoupledSpinRotorAtom, HalfI32> {
        let l_max = particles.get::<RotorLMax>()
            .expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>()
            .expect("Did not find RotorJMax parameter in particles").0;

        let l_states = (0..=l_max)
            .map(|l| {
                let l = HalfU32::from_doubled(2 * l);
                let projections = spin_projections(l);
                State::new(UncoupledSpinRotorAtom::SystemL(l), projections)
            })
            .collect();

        let n_states = (0..=j_max)
            .map(|j| {
                let j = HalfU32::from_doubled(2 * j);
                let projections = spin_projections(j);
                State::new(UncoupledSpinRotorAtom::RotorN(j), projections)
            })
            .collect();

        let s_rotor = State::new(
            UncoupledSpinRotorAtom::RotorS(self.hifi_problem.first.s),
            spin_projections(self.hifi_problem.first.s),
        );
        let s_atom = State::new(
            UncoupledSpinRotorAtom::AtomS(self.hifi_problem.second.s),
            spin_projections(self.hifi_problem.second.s),
        );
        let i_rotor = State::new(
            UncoupledSpinRotorAtom::RotorI(self.hifi_problem.first.i),
            spin_projections(self.hifi_problem.first.i),
        );
        let i_atom = State::new(
            UncoupledSpinRotorAtom::AtomI(self.hifi_problem.second.i),
            spin_projections(self.hifi_problem.second.i),
        );

        let mut states = States::default();
        states.push_state(StateType::Sum(l_states))
            .push_state(StateType::Sum(n_states))
            .push_state(StateType::Irreducible(s_rotor))
            .push_state(StateType::Irreducible(i_rotor))
            .push_state(StateType::Irreducible(s_atom))
            .push_state(StateType::Irreducible(i_atom));

        let mut basis = match self.hifi_problem.total_projection {
            Some(m_tot) => {
                states.iter_elements()
                    .filter(|els| {
                        els.values.iter().copied().sum::<HalfI32>() == m_tot
                    })
                    .collect()
            },
            None => states.get_basis(),
        };

        basis.sort_by_key(|s| {
            cast_variant!(s.variants[0], UncoupledSpinRotorAtom::SystemL)
        });

        basis
    }
}

#[inline]
pub fn p3_factor(s: &Spin) -> f64 {
    let s = s.s.value();
    ((2. * s + 1.) * s * (s + 1.)).sqrt()
}

pub struct UncoupledAlkaliRotorAtomProblem<P: SimplePotential, V: SimplePotential> {
    pub basis: StatesBasis<UncoupledSpinRotorAtom, HalfI32>,
    pub angular_blocks: AngularBlocks,
    triplets: Vec<(P, Mat<f64>)>,
    singlets: Vec<(V, Mat<f64>)>,
}

impl<P, V> UncoupledAlkaliRotorAtomProblem<P, V> 
where 
    P: SimplePotential + Clone,
    V: SimplePotential + Clone
{
    pub fn scattering_at_field(&self, mag_field: f64) -> ScatteringProblem<impl MatPotential, impl BasisDescription> {
        let (energies, states) = self.angular_blocks.diagonalize(mag_field);

        let energy_levels = energies.iter()
            .map(|e| Dispersion::new(*e, 0))
            .collect();

        let energy_levels = Diagonal::from_vec(energy_levels);

        let mut triplets_iter = self.triplets.iter()
            .map(|(p, m)| {
                let masking = states.transpose() * m * states.as_ref();
                MaskedPotential::new(p.clone(), masking)
            });
        let mut triplets = Composite::new(triplets_iter.next().expect("No triplet potentials found"));
        for p in triplets_iter {
            triplets.add_potential(p);
        }

        let mut singlets_iter = self.singlets.iter()
            .map(|(p, m)| {
                let masking = states.transpose() * m * states.as_ref();
                MaskedPotential::new(p.clone(), masking)
            });
        let mut singlets = Composite::new(singlets_iter.next().expect("No triplet potentials found"));
        for p in singlets_iter {
            singlets.add_potential(p);
        }

        let potential = PairPotential::new(triplets, singlets);
        let full_potential = PairPotential::new(energy_levels, potential);

        let asymptotic = Asymptotic {
            channel_energies: energies,
            centrifugal: self.angular_blocks.angular_states(),
            entrance: 0,
            channel_states: states,
        };
        
        ScatteringProblem {
            potential: full_potential,
            asymptotic,
            basis_description: IndexBasisDescription
        }
    }

    pub fn levels_at_field(&self, l: u32, mag_field: f64) -> (Vec<f64>, Mat<f64>) {
        let block = &self.angular_blocks.0[l as usize];
        let internal = &block.field_inv + mag_field * &block.field_prop;

        diagonalize(internal.as_ref())
    }
}
