use abm::{get_hifi, get_zeeman_prop, utility::diagonalize, DoubleHifiProblemBuilder, Symmetry};
use clebsch_gordan::{half_i32, half_integer::{HalfI32, HalfU32}, half_u32, wigner_3j, wigner_6j};
use faer::Mat;
use quantum::{cast_variant, params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, spins::spin_projections, state::State, state_type::StateType, States, StatesBasis}};
use scattering_solver::{boundary::Asymptotic, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::{Potential, SimplePotential}}, utility::AngMomentum};

use crate::{utility::{AnisoHifi, GammaSpinRot, RotorJMax, RotorDoubleJTotMax, RotorLMax}, ScatteringProblem};

#[derive(Clone)]
pub struct AlkaliRotorAtomProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    hifi_problem: DoubleHifiProblemBuilder,
    triplet_potential: Vec<(u32, P)>,
    singlet_potential: Vec<(u32, V)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinRotorAtom {
    /// (l, j, j_tot)
    Angular((u32, u32, u32)),
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
    pub fn new(hifi_problem: DoubleHifiProblemBuilder, triplet: Vec<(u32, P)>, singlet: Vec<(u32, V)>) -> Self {
        assert!(hifi_problem.first.s == half_u32!(1/2));
        assert!(hifi_problem.second.s == half_u32!(1/2));
        assert!(matches!(hifi_problem.symmetry, Symmetry::None));

        Self {
            hifi_problem,
            triplet_potential: triplet,
            singlet_potential: singlet,
        }
    }

    pub fn build(self, particles: &Particles) -> AlkaliRotorAtomProblem<P, V> {
        let l_max = particles.get::<RotorLMax>().expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>().expect("Did not find RotorJMax parameter in particles").0;
        let j_tot_max = particles.get::<RotorDoubleJTotMax>().map_or(0, |x| x.0);
        // todo! change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>().expect("Did not find RotConst parameter in the particles").0;
        let gamma_spin_rot = particles.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = particles.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;

        let ls: Vec<u32> = (0..=l_max).collect();
        
        let mut angular_states = vec![];
        for j_tot in 0..=j_tot_max {
            for &l in &ls {
                let j_lower = (l as i32 - j_tot as i32).unsigned_abs().min(j_max);
                let j_upper = (l + j_tot).min(j_max);
                
                for j in j_lower..=j_upper {
                    let projections = spin_projections(HalfU32::from_doubled(2 * j_tot));
                    let state = State::new(SpinRotorAtom::Angular((l, j, j_tot)), projections);
                    angular_states.push(state);
                }
            }
        }

        let s_rotor = State::new(
            SpinRotorAtom::RotorS(self.hifi_problem.first.s),
            spin_projections(self.hifi_problem.first.s),
        );
        let s_atom = State::new(
            SpinRotorAtom::AtomS(self.hifi_problem.second.s),
            spin_projections(self.hifi_problem.second.s),
        );
        let i_rotor = State::new(
            SpinRotorAtom::RotorI(self.hifi_problem.first.i),
            spin_projections(self.hifi_problem.first.i),
        );
        let i_atom = State::new(
            SpinRotorAtom::AtomI(self.hifi_problem.second.i),
            spin_projections(self.hifi_problem.second.i),
        );

        let mut states = States::default();

        states.push_state(StateType::Sum(angular_states))
            .push_state(StateType::Irreducible(s_rotor))
            .push_state(StateType::Irreducible(i_rotor))
            .push_state(StateType::Irreducible(s_atom))
            .push_state(StateType::Irreducible(i_atom));

        let basis = match self.hifi_problem.total_projection {
            Some(m_tot) => {
                states.iter_elements()
                    .filter(|els| {
                        els.values.iter().copied().sum::<HalfI32>() == m_tot
                    })
                    .collect()
            },
            None => states.get_basis(),
        };

        let j_centrifugal = Operator::from_diagonal_mel(&basis, [SpinRotorAtom::Angular((0, 0, 0))], |[ang]| {
            let (_, j, _) = cast_variant!(ang.0, SpinRotorAtom::Angular);

            rot_const * (j * (j + 1)) as f64
        });

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.hifi_problem.first.a_hifi {
            hifi += get_hifi!(basis, SpinRotorAtom::RotorS, SpinRotorAtom::RotorI, a_hifi).as_ref()
        }
        if let Some(a_hifi) = self.hifi_problem.second.a_hifi {
            hifi += get_hifi!(basis, SpinRotorAtom::AtomS, SpinRotorAtom::AtomI, a_hifi).as_ref()
        }

        let mut zeeman_prop = get_zeeman_prop!(basis, SpinRotorAtom::RotorS, self.hifi_problem.first.gamma_e).as_ref()
            + get_zeeman_prop!(basis, SpinRotorAtom::AtomS, self.hifi_problem.second.gamma_e).as_ref();
        if let Some(gamma_i) = self.hifi_problem.first.gamma_i {
            zeeman_prop += get_zeeman_prop!(basis, SpinRotorAtom::RotorI, gamma_i).as_ref()
        }
        if let Some(gamma_i) = self.hifi_problem.second.gamma_i {
            zeeman_prop += get_zeeman_prop!(basis, SpinRotorAtom::AtomI, gamma_i).as_ref()
        }

        let spin_rot = Operator::from_mel(
            &basis, 
            [SpinRotorAtom::Angular((0, 0, 0)), SpinRotorAtom::RotorS(half_u32!(0))],
            |[ang_braket, s_braket]| {
                let (l_ket, j_ket, j_tot_ket) = cast_variant!(ang_braket.ket.0, SpinRotorAtom::Angular);
                let m_r_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = cast_variant!(ang_braket.bra.0, SpinRotorAtom::Angular);
                let m_r_bra = ang_braket.bra.1;

                let s_ket = cast_variant!(s_braket.ket.0, SpinRotorAtom::RotorS);
                let ms_ket = s_braket.ket.1;
                let s_bra = cast_variant!(s_braket.bra.0, SpinRotorAtom::RotorS);
                let ms_bra = s_braket.bra.1;

                if l_bra == l_ket && j_bra == j_ket && s_ket == s_bra {
                    let factor = (((2 * j_tot_ket + 1) * (2 * j_tot_bra + 1) 
                                * (2 * j_ket + 1) * j_ket * (j_ket + 1)) as f64).sqrt()
                                * ((2. * s_ket.value() + 1.) * s_ket.value() * (s_ket.value() + 1.)).sqrt();

                    let sign = (-1.0f64).powi(1 + (j_tot_bra + j_tot_ket + l_bra + j_bra) as i32 - m_r_bra.double_value() / 2
                                                + (s_bra.double_value() as i32 - ms_bra.double_value()) / 2);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let j_tot_ket = HalfU32::from_doubled(2 * j_tot_ket);
                    
                    let mut wigner_sum = 0.;
                    for p in [half_i32!(-1), half_i32!(0), half_i32!(1)] { 
                        wigner_sum += (-1.0f64).powi(p.double_value() / 2) 
                            * wigner_6j(j_bra, j_tot_bra, l_bra, j_tot_ket, j_bra, half_u32!(1/2))
                            * wigner_3j(j_tot_bra, half_u32!(1), j_tot_ket, -m_r_bra, p, m_r_ket)
                            * wigner_3j(s_bra, half_u32!(1), s_bra, -ms_bra, p, ms_ket)
                    }

                    gamma_spin_rot * factor * sign * wigner_sum
                } else {
                    0.
                }
            }
        );

        let aniso_hifi = Operator::from_mel(
            &basis, 
            [SpinRotorAtom::Angular((0, 0, 0)), SpinRotorAtom::RotorS(half_u32!(0)), SpinRotorAtom::RotorI(half_u32!(0))],
            |[ang_braket, s_braket, i_braket]| {
                let (l_ket, j_ket, j_tot_ket) = cast_variant!(ang_braket.ket.0, SpinRotorAtom::Angular);
                let mr_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = cast_variant!(ang_braket.bra.0, SpinRotorAtom::Angular);
                let mr_bra = ang_braket.bra.1;

                let s_ket = cast_variant!(s_braket.ket.0, SpinRotorAtom::RotorS);
                let ms_ket = s_braket.ket.1;
                let s_bra = cast_variant!(s_braket.bra.0, SpinRotorAtom::RotorS);
                let ms_bra = s_braket.bra.1;

                let i_ket = cast_variant!(i_braket.ket.0, SpinRotorAtom::RotorI);
                let mi_ket = i_braket.ket.1;
                let i_bra = cast_variant!(i_braket.bra.0, SpinRotorAtom::RotorI);
                let mi_bra = i_braket.bra.1;

                if l_bra == l_ket && s_ket == s_bra && i_ket == i_bra {
                    let factor = (((2 * j_tot_ket + 1) * (2 * j_tot_bra + 1)
                                    * (2 * j_ket + 1) * (2 * j_bra + 1)) as f64).sqrt()
                                * ((2. * s_bra.value() + 1.) * s_bra.value() * (s_bra.value() + 1.)
                                    * (2. * i_bra.value() + 1.) * i_bra.value() * (i_bra.value() + 1.)).sqrt();

                    let sign = (-1.0f64).powi((j_tot_bra + j_tot_ket + l_bra) as i32 - mr_bra.double_value() / 2);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let j_ket = HalfU32::from_doubled(2 * j_ket);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let j_tot_ket = HalfU32::from_doubled(2 * j_tot_ket);

                    let wigner = wigner_6j(j_bra, j_tot_bra, l_bra, j_tot_ket, j_ket, half_u32!(1))
                        * wigner_3j(half_u32!(1), half_u32!(1), half_u32!(2), mi_bra - mi_ket, ms_bra - ms_ket, mr_bra - mr_ket)
                        * wigner_3j(j_tot_bra, half_u32!(2), j_tot_ket, -mr_bra, mr_bra - mr_ket, mr_ket)
                        * wigner_3j(j_bra, half_u32!(2), j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                        * wigner_3j(i_bra, half_u32!(1), i_ket, -mi_bra, mi_bra - mi_ket, mi_ket)
                        * wigner_3j(s_bra, half_u32!(1), s_ket, -ms_bra, ms_bra - ms_ket, ms_ket);

                    aniso_hifi * f64::sqrt(30.) / 3. * sign * factor * wigner
                } else {
                    0.
                }
            }
        );

        let singlet_potentials = self.singlet_potential.into_iter()
            .map(|(lambda, pot)| {
                let masking_singlet = Operator::from_mel(
                    &basis, 
                    [SpinRotorAtom::Angular((0, 0, 0)), SpinRotorAtom::RotorS(half_u32!(0)), SpinRotorAtom::AtomS(half_u32!(0))],
                    |[ang_braket, s_braket, sa_braket]| {
                        let (l_ket, j_ket, j_tot_ket) = cast_variant!(ang_braket.ket.0, SpinRotorAtom::Angular);
                        let mr_ket = ang_braket.ket.1;
        
                        let (l_bra, j_bra, j_tot_bra) = cast_variant!(ang_braket.bra.0, SpinRotorAtom::Angular);
                        let mr_bra = ang_braket.bra.1;
        
                        let s_ket = cast_variant!(s_braket.ket.0, SpinRotorAtom::RotorS);
                        let ms_ket = s_braket.ket.1;
                        let s_bra = cast_variant!(s_braket.bra.0, SpinRotorAtom::RotorS);
                        let ms_bra = s_braket.bra.1;
        
                        let sa_ket = cast_variant!(sa_braket.ket.0, SpinRotorAtom::AtomS);
                        let msa_ket = sa_braket.ket.1;
                        let sa_bra = cast_variant!(sa_braket.bra.0, SpinRotorAtom::AtomS);
                        let msa_bra = sa_braket.bra.1;
        
                        if j_tot_bra == j_tot_ket && mr_bra == mr_ket && s_ket == s_bra && sa_ket == sa_bra {
                            let factor = (((2 * j_bra + 1) * (2 * j_ket + 1)
                                        * (2 * l_bra + 1) * (2 * l_ket + 1)) as f64).sqrt();
        
                            let sign = (-1.0f64).powi((j_tot_bra + j_bra + j_ket + s_bra.double_value()) as i32 - sa_bra.double_value() as i32);
        
                            let j_bra = HalfU32::from_doubled(2 * j_bra);
                            let j_ket = HalfU32::from_doubled(2 * j_ket);
                            let l_bra = HalfU32::from_doubled(2 * l_bra);
                            let l_ket = HalfU32::from_doubled(2 * l_ket);
                            let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                            let lambda = HalfU32::from_doubled(lambda);

                            let wigner = wigner_6j(j_bra, l_bra, j_tot_bra, l_ket, j_ket, lambda)
                                * wigner_3j(j_bra, lambda, j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                                * wigner_3j(l_bra, lambda, l_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                                * wigner_3j(s_bra, sa_bra, half_u32!(0), ms_bra, msa_bra, half_i32!(0))
                                * wigner_3j(s_bra, sa_bra, half_u32!(0), ms_ket, msa_ket, half_i32!(0));
        
                            sign * factor * wigner
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
                let masking_triplet = Operator::from_mel(
                    &basis, 
                    [SpinRotorAtom::Angular((0, 0, 0)), SpinRotorAtom::RotorS(half_u32!(0)), SpinRotorAtom::AtomS(half_u32!(0))],
                    |[ang_braket, s_braket, sa_braket]| {
                        let (l_ket, j_ket, j_tot_ket) = cast_variant!(ang_braket.ket.0, SpinRotorAtom::Angular);
                        let mr_ket = ang_braket.ket.1;
        
                        let (l_bra, j_bra, j_tot_bra) = cast_variant!(ang_braket.bra.0, SpinRotorAtom::Angular);
                        let mr_bra = ang_braket.bra.1;
        
                        let s_ket = cast_variant!(s_braket.ket.0, SpinRotorAtom::RotorS);
                        let ms_ket = s_braket.ket.1;
                        let s_bra = cast_variant!(s_braket.bra.0, SpinRotorAtom::RotorS);
                        let ms_bra = s_braket.bra.1;
        
                        let sa_ket = cast_variant!(sa_braket.ket.0, SpinRotorAtom::AtomS);
                        let msa_ket = sa_braket.ket.1;
                        let sa_bra = cast_variant!(sa_braket.bra.0, SpinRotorAtom::AtomS);
                        let msa_bra = sa_braket.bra.1;
        
                        if j_tot_bra == j_tot_ket && mr_bra == mr_ket && s_ket == s_bra && sa_ket == sa_bra {
                            let factor = (((2 * j_bra + 1) * (2 * j_ket + 1)
                                        * (2 * l_bra + 1) * (2 * l_ket + 1)) as f64).sqrt();
        
                            let sign = (-1.0f64).powi((j_tot_bra + j_bra + j_ket + s_bra.double_value()) as i32 - sa_bra.double_value() as i32);
        
                            let j_bra = HalfU32::from_doubled(2 * j_bra);
                            let j_ket = HalfU32::from_doubled(2 * j_ket);
                            let l_bra = HalfU32::from_doubled(2 * l_bra);
                            let l_ket = HalfU32::from_doubled(2 * l_ket);
                            let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                            let lambda = HalfU32::from_doubled(lambda);

                            let mut triplet_wigner = 0.;
                            for ms_tot in [half_i32!(-1), half_i32!(0), half_i32!(1)] {
                               triplet_wigner += 3. * (-1.0f64).powi(ms_tot.double_value() / 2) 
                                * wigner_3j(s_bra, sa_bra, half_u32!(1), ms_bra, msa_bra, -ms_tot)
                                * wigner_3j(s_bra, sa_bra, half_u32!(1), ms_ket, msa_ket, -ms_tot)
                            }
    
                            let wigner = wigner_6j(j_bra, l_bra, j_tot_bra, l_ket, j_ket, lambda)
                                * wigner_3j(j_bra, lambda, j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                                * wigner_3j(l_bra, lambda, l_ket, half_i32!(0), half_i32!(0), half_i32!(0));
        
                            sign * factor * wigner * triplet_wigner
                        } else {
                            0.
                        }
                    }
                );

                (pot, masking_triplet.into_backed())
            })
            .collect();

        AlkaliRotorAtomProblem {
            basis,
            triplets: triplet_potentials,
            singlets: singlet_potentials,
            mag_inv: hifi + aniso_hifi.as_ref() + j_centrifugal.as_ref() + spin_rot.as_ref(),
            mag_prop: zeeman_prop,
        }
    }
}

pub struct AlkaliRotorAtomProblem<P: SimplePotential, V: SimplePotential> {
    basis: StatesBasis<SpinRotorAtom, HalfI32>,
    triplets: Vec<(P, Mat<f64>)>,
    singlets: Vec<(V, Mat<f64>)>,
    mag_inv: Mat<f64>,
    mag_prop: Mat<f64>
}

impl<P, V> AlkaliRotorAtomProblem<P, V> 
where 
    P: SimplePotential + Clone,
    V: SimplePotential + Clone
{
    pub fn scattering_at_field(&self, mag_field: f64, entrance: usize) -> ScatteringProblem<impl Potential<Space = Mat<f64>>> {
        let internal = &self.mag_inv + mag_field * &self.mag_prop;

        let (energies, states) = diagonalize(internal.as_ref());
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

        let l = Operator::from_diagonal_mel(&self.basis, [SpinRotorAtom::Angular((0, 0, 0))], |[a]| {
            let (l, _, _) = cast_variant!(a.0, SpinRotorAtom::Angular);

            l as f64
        }).into_backed();

        let l_diag = states.transpose() * l * states.as_ref();
        let centrifugal = l_diag.diagonal()
            .column_vector()
            .iter()
            .map(|l| AngMomentum(l.round() as u32))
            .collect();

        let asymptotic = Asymptotic {
            channel_energies: energies,
            centrifugal,
            entrance,
            channel_states: states,
        };
        
        ScatteringProblem {
            potential: full_potential,
            asymptotic,
        }
    }

    pub fn levels_at_field(&self, mag_field: f64) -> (Vec<f64>, Mat<f64>) {
        let internal = &self.mag_inv + mag_field * &self.mag_prop;

        diagonalize(internal.as_ref())
    }
}