use abm::{get_hifi, get_zeeman_prop, utility::diagonalize, DoubleHifiProblemBuilder, Symmetry};
use clebsch_gordan::{half_i32, half_integer::{HalfI32, HalfU32}, half_u32};
use faer::Mat;
use quantum::{cast_variant, params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, spins::spin_projections, state::State, state_type::StateType, States, StatesBasis}};
use scattering_solver::{boundary::Asymptotic, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::{MatPotential, SimplePotential}}, utility::AngMomentum};

use crate::{angular_block::{AngularBlock, AngularBlocks}, get_aniso_hifi, get_rotor_atom_potential_masking, get_spin_rot, utility::{AnisoHifi, GammaSpinRot, RotorJMax, RotorJTotMax, RotorLMax}, BasisDescription, IndexBasisDescription, ScatteringProblem};

#[derive(Clone)]
pub struct AlkaliRotorAtomProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub(super) hifi_problem: DoubleHifiProblemBuilder,
    pub(super) triplet_potential: Vec<(u32, P)>,
    pub(super) singlet_potential: Vec<(u32, V)>,
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
        assert!(hifi_problem.second.s == half_u32!(1/2)
            || hifi_problem.second.s == half_u32!(0));
        assert!(matches!(hifi_problem.symmetry, Symmetry::None));

        Self {
            hifi_problem,
            triplet_potential: triplet,
            singlet_potential: singlet,
        }
    }

    pub fn build(self, particles: &Particles) -> AlkaliRotorAtomProblem<P, V> {
        // todo! change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>()
            .expect("Did not find RotConst parameter in the particles").0;
        let gamma_spin_rot = particles.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = particles.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;

        let l_max = particles.get::<RotorLMax>()
            .expect("Did not find SystemLMax parameter in particles").0;
        let ordered_basis = self.basis(particles);

        assert!(ordered_basis.is_sorted_by_key(|s| cast_variant!(s.variants[0], SpinRotorAtom::Angular).0));

        let angular_block_basis = (0..=l_max).map(|l| {
                let red_basis = ordered_basis.iter().filter(|s| {
                    matches!(s.variants[0], SpinRotorAtom::Angular((l_current, _, _)) if l == l_current)
                })
                .cloned()
                .collect::<StatesBasis<SpinRotorAtom, HalfI32>>();

                (l, red_basis)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect::<Vec<(u32, StatesBasis<SpinRotorAtom, HalfI32>)>>();

        let angular_blocks = angular_block_basis.iter()
            .map(|(l, basis)| {
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

                let spin_rot = get_spin_rot!(&basis, SpinRotorAtom::Angular, SpinRotorAtom::RotorS, gamma_spin_rot);

                let aniso_hifi = get_aniso_hifi!(&basis, SpinRotorAtom::Angular, 
                    SpinRotorAtom::RotorS, SpinRotorAtom::RotorI, aniso_hifi);

                AngularBlock {
                    ang_momentum: AngMomentum(*l),
                    field_inv: hifi + aniso_hifi.as_ref() + j_centrifugal.as_ref() + spin_rot.as_ref(),
                    field_prop: zeeman_prop,
                }
            })
            .collect();

        let singlet_potentials = self.singlet_potential.into_iter()
            .map(|(lambda, pot)| {
                let masking_singlet = get_rotor_atom_potential_masking!(
                    Singlet lambda; &ordered_basis, SpinRotorAtom::Angular, 
                    SpinRotorAtom::RotorS, SpinRotorAtom::AtomS
                );

                (pot, masking_singlet.into_backed())
            })
            .collect();

        let triplet_potentials = self.triplet_potential.into_iter()
            .map(|(lambda, pot)| {
                let masking_triplet = get_rotor_atom_potential_masking!(
                    Triplet lambda; &ordered_basis, SpinRotorAtom::Angular, 
                    SpinRotorAtom::RotorS, SpinRotorAtom::AtomS
                );

                (pot, masking_triplet.into_backed())
            })
            .collect();

        AlkaliRotorAtomProblem {
            angular_blocks: AngularBlocks(angular_blocks),
            basis: ordered_basis,
            triplets: triplet_potentials,
            singlets: singlet_potentials,
        }
    }

    fn basis(&self, particles: &Particles) -> StatesBasis<SpinRotorAtom, HalfI32> {
        let l_max = particles.get::<RotorLMax>()
            .expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>()
            .expect("Did not find RotorJMax parameter in particles").0;
        let j_tot_max = particles.get::<RotorJTotMax>().map_or(0, |x| x.0);
            
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
            let (l, _, _) = cast_variant!(s.variants[0], SpinRotorAtom::Angular);

            l
        });

        basis
    }
}

pub struct AlkaliRotorAtomProblem<P: SimplePotential, V: SimplePotential> {
    pub basis: StatesBasis<SpinRotorAtom, HalfI32>,
    pub angular_blocks: AngularBlocks,
    triplets: Vec<(P, Mat<f64>)>,
    singlets: Vec<(V, Mat<f64>)>,
}

impl<P, V> AlkaliRotorAtomProblem<P, V> 
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
