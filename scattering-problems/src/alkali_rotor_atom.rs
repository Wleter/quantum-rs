use abm::{utility::diagonalize, DoubleHifiProblemBuilder, Symmetry};
use clebsch_gordan::hu32;
use faer::Mat;
use quantum::{cast_variant, operator_diagonal_mel, operator_mel, params::{particle_factory::RotConst, Params}, states::{operator::Operator, spins::{get_spin_basis, Spin, SpinOperators}, state::{into_variant, StateBasis}, States, StatesBasis}};
use scattering_solver::{boundary::Asymptotic, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::{MatPotential, SimplePotential}}, utility::AngMomentum};

use crate::{angular_block::{AngularBlock, AngularBlocks}, utility::{aniso_hifi_tram_mel, create_angular_pairs, percival_coef_tram_mel, singlet_projection_uncoupled, spin_rot_tram_mel, triplet_projection_uncoupled, AngularPair, AnisoHifi, GammaSpinRot}, FieldScatteringProblem, IndexBasisDescription, ScatteringProblem};

#[derive(Clone)]
pub struct AlkaliRotorAtomProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub singlet_potential: Vec<(u32, V)>,
    pub triplet_potential: Vec<(u32, P)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TramStates {
    Angular(AngularPair),
    NTot(Spin),
    RotorS(Spin),
    RotorI(Spin),
    AtomS(Spin),
    AtomI(Spin)
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ParityBlock {
    #[default]
    Positive,
    Negative,
    All
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TramBasisRecipe {
    pub l_max: u32,
    pub n_max: u32,
    pub n_tot_max: u32,
    pub parity: ParityBlock
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UncoupledRotorBasisRecipe {
    pub l_max: u32,
    pub n_max: u32,
    pub parity: ParityBlock
}

impl<P, V> AlkaliRotorAtomProblemBuilder<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub fn new(triplet: Vec<(u32, P)>, singlet: Vec<(u32, V)>) -> Self {
        Self {
            triplet_potential: triplet,
            singlet_potential: singlet,
        }
    }

    pub fn build(self, params: &Params, basis_recipe: &TramBasisRecipe) -> AlkaliRotorAtomProblem<TramStates, P, V> {
        let rot_const = params.get::<RotConst>()
            .expect("Did not find RotConst parameter in the params").0;
        let gamma_spin_rot = params.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = params.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;
        let hifi_problem = params.get::<DoubleHifiProblemBuilder>()
            .expect("Did not find DoubleHifiProblemBuilder in the params");

        assert!(hifi_problem.first.s == hu32!(1/2));
        assert!(hifi_problem.second.s == hu32!(1/2)
            || hifi_problem.second.s == hu32!(0));
        assert!(matches!(hifi_problem.symmetry, Symmetry::None));

        let l_max = basis_recipe.l_max;
        let ordered_basis = self.basis(hifi_problem, basis_recipe);

        assert!(ordered_basis.is_sorted_by_key(|s| cast_variant!(s[0], TramStates::Angular).l));

        let angular_block_basis = (0..=l_max).map(|l| {
                let red_basis = ordered_basis.iter().filter(|s| {
                    matches!(s[0], TramStates::Angular(ang_curr) if ang_curr.l == l)
                })
                .cloned()
                .collect::<StatesBasis<TramStates>>();

                (l, red_basis)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect::<Vec<(u32, StatesBasis<TramStates>)>>();

        let angular_blocks = angular_block_basis.iter()
            .map(|(l, basis)| {
                let n_centrifugal = operator_diagonal_mel!(&basis, |[ang: TramStates::Angular]| {
                    rot_const * ang.n.value() * (ang.n.value() + 1.0)
                });
        
                let mut hifi = Mat::zeros(basis.len(), basis.len());
                if let Some(a_hifi) = hifi_problem.first.a_hifi {
                    hifi += operator_mel!(&basis, |[s: TramStates::RotorS, i: TramStates::RotorI]| {
                        a_hifi * SpinOperators::dot(s, i)
                    }).as_ref();
                }
                if let Some(a_hifi) = hifi_problem.second.a_hifi {
                    hifi += operator_mel!(&basis, |[s: TramStates::AtomS, i: TramStates::AtomI]| {
                        a_hifi * SpinOperators::dot(s, i)
                    }).as_ref();
                }
        
                let mut zeeman_prop = operator_diagonal_mel!(&basis, |[s_rotor: TramStates::RotorS, s_atom: TramStates::AtomS]| {
                    -hifi_problem.first.gamma_e * s_rotor.ms.value() - hifi_problem.second.gamma_e * s_atom.ms.value()
                }).into_backed();
                if let Some(gamma_i) = hifi_problem.first.gamma_i {
                    zeeman_prop += operator_diagonal_mel!(&basis, |[i: TramStates::RotorI]| {
                        -gamma_i * i.ms.value()
                    }).as_ref()
                }
                if let Some(gamma_i) = hifi_problem.second.gamma_i {
                    zeeman_prop += operator_diagonal_mel!(&basis, |[i: TramStates::AtomI]| {
                        -gamma_i * i.ms.value()
                    }).as_ref()
                }

                let spin_rot = operator_mel!(&basis, 
                    |[ang: TramStates::Angular, n_tot: TramStates::NTot, s: TramStates::RotorS]| {
                        gamma_spin_rot * spin_rot_tram_mel(ang, n_tot, s)
                    }
                );

                let aniso_hifi = operator_mel!(&basis, 
                    |[ang: TramStates::Angular, n_tot: TramStates::NTot, s: TramStates::RotorS, i: TramStates::RotorI]| {
                        aniso_hifi * aniso_hifi_tram_mel(ang, n_tot, s, i)
                    }
                );

                let field_inv = vec![
                    hifi, 
                    aniso_hifi.into_backed(),
                    n_centrifugal.into_backed(),
                    spin_rot.into_backed()
                ];

                let field_prop = vec![zeeman_prop];

                AngularBlock::new(AngMomentum(*l), field_inv, field_prop)
            })
            .collect();

        let singlet_potentials = self.singlet_potential.into_iter()
            .map(|(lambda, pot)| {
                let masking_singlet = operator_mel!(&ordered_basis, 
                    |[ang: TramStates::Angular, n_tot: TramStates::NTot, s_rotor: TramStates::RotorS, s_atom: TramStates::AtomS]| {
                        singlet_projection_uncoupled(s_rotor, s_atom) * percival_coef_tram_mel(lambda, ang, n_tot)
                    }
                );

                (pot, masking_singlet.into_backed())
            })
            .collect();

        let triplet_potentials = self.triplet_potential.into_iter()
            .map(|(lambda, pot)| {
                let masking_triplet = operator_mel!(&ordered_basis, 
                    |[ang: TramStates::Angular, n_tot: TramStates::NTot, s_rotor: TramStates::RotorS, s_atom: TramStates::AtomS]| {
                        triplet_projection_uncoupled(s_rotor, s_atom) * percival_coef_tram_mel(lambda, ang, n_tot)
                    }
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

    fn basis(&self, hifi_problem: &DoubleHifiProblemBuilder, basis_recipe: &TramBasisRecipe) -> StatesBasis<TramStates> {
        let l_max = basis_recipe.l_max;
        let n_max = basis_recipe.n_max;
        let n_tot_max = basis_recipe.n_tot_max;
        let parity = basis_recipe.parity;
        
        let angular_states = create_angular_pairs(l_max, n_max, n_tot_max, parity);
        let angular_states = into_variant(angular_states, TramStates::Angular);

        let total_angular = (0..=n_tot_max).map(|n_tot| {
            into_variant(get_spin_basis(n_tot.into()), TramStates::NTot)
        })
        .flatten()
        .collect();

        let s_rotor = into_variant(get_spin_basis(hifi_problem.first.s), TramStates::RotorS);
        let s_atom = into_variant(get_spin_basis(hifi_problem.second.s), TramStates::AtomS);
        let i_rotor = into_variant(get_spin_basis(hifi_problem.first.i), TramStates::RotorI);
        let i_atom = into_variant(get_spin_basis(hifi_problem.second.i), TramStates::AtomI);

        let mut states = States::default();
        states.push_state(StateBasis::new(angular_states))
            .push_state(StateBasis::new(total_angular))
            .push_state(StateBasis::new(s_rotor))
            .push_state(StateBasis::new(s_atom))
            .push_state(StateBasis::new(i_rotor))
            .push_state(StateBasis::new(i_atom));

        let mut basis: StatesBasis<TramStates> = match hifi_problem.total_projection {
            Some(m_tot) => {
                states.iter_elements()
                    .filter(|b| {
                        let ang = cast_variant!(b[0], TramStates::Angular);
                        let m_n_tot = cast_variant!(b[1], TramStates::NTot);
                        let s_rotor = cast_variant!(b[2], TramStates::RotorS);
                        let s_atom = cast_variant!(b[3], TramStates::AtomS);
                        let i_rotor = cast_variant!(b[4], TramStates::RotorI);
                        let i_atom = cast_variant!(b[5], TramStates::AtomI);

                        m_n_tot.ms + s_rotor.ms + s_atom.ms + i_rotor.ms + i_atom.ms == m_tot
                            && (ang.l + ang.n) >= m_n_tot.s
                            && (ang.l + m_n_tot.s) >= ang.n 
                            && (ang.n + m_n_tot.s) >= ang.l
                    })
                    .collect()
            },
            None => states.iter_elements()
                .filter(|b| {
                    let ang = cast_variant!(b[0], TramStates::Angular);
                    let m_n_tot = cast_variant!(b[1], TramStates::NTot);
                    
                    (ang.l + ang.n) >= m_n_tot.s
                        && (ang.l + m_n_tot.s) >= ang.n 
                        && (ang.n + m_n_tot.s) >= ang.l
                })
                .collect()
        };

        basis.sort_by_key(|b| cast_variant!(b[0], TramStates::Angular).l);

        basis
    }
}

pub struct AlkaliRotorAtomProblem<T, P: SimplePotential, V: SimplePotential> {
    pub basis: StatesBasis<T>,
    pub angular_blocks: AngularBlocks,
    pub(super) triplets: Vec<(P, Mat<f64>)>,
    pub(super) singlets: Vec<(V, Mat<f64>)>,
}

impl<T, P, V> FieldScatteringProblem<IndexBasisDescription> for AlkaliRotorAtomProblem<T, P, V> 
where 
    P: SimplePotential + Clone,
    V: SimplePotential + Clone
{
    fn scattering_for(&self, mag_field: f64) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
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

    fn levels(&self, field: f64, l: Option<u32>) -> (Vec<f64>, Mat<f64>) {
        if let Some(l) = l {
            let block = &self.angular_blocks.0[l as usize];
            let internal = &block.field_inv() + field * &block.field_prop();
    
            diagonalize(internal.as_ref())
        } else {
            self.angular_blocks.diagonalize(field)
        }
    }
}
