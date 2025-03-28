use abm::DoubleHifiProblemBuilder;
use faer::Mat;
use quantum::{
    cast_variant, operator_diagonal_mel, operator_mel,
    params::{Params, particle_factory::RotConst},
    states::{
        States, StatesBasis,
        spins::{Spin, SpinOperators, get_spin_basis},
        state::{StateBasis, into_variant},
    },
};
use scattering_solver::{potentials::potential::SimplePotential, utility::AngMomentum};

use crate::{
    alkali_rotor_atom::{
        AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder, ParityBlock,
        UncoupledRotorBasisRecipe,
    },
    angular_block::{AngularBlock, AngularBlocks},
    utility::{
        AnisoHifi, GammaSpinRot, aniso_hifi_uncoupled_mel, percival_coef_uncoupled_mel,
        singlet_projection_uncoupled, triplet_projection_uncoupled,
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UncoupledAlkaliRotorAtomStates {
    SystemL(Spin),
    RotorN(Spin),
    RotorS(Spin),
    RotorI(Spin),
    AtomS(Spin),
    AtomI(Spin),
}

impl<P, V> AlkaliRotorAtomProblemBuilder<P, V>
where
    P: SimplePotential,
    V: SimplePotential,
{
    pub fn build_uncoupled(
        self,
        params: &Params,
        basis_recipe: &UncoupledRotorBasisRecipe,
    ) -> AlkaliRotorAtomProblem<UncoupledAlkaliRotorAtomStates, P, V> {
        use UncoupledAlkaliRotorAtomStates::*;

        let rot_const = params
            .get::<RotConst>()
            .expect("Did not find RotConst parameter in the params")
            .0;
        let gamma_spin_rot = params.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = params.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;
        let hifi_problem = params
            .get::<DoubleHifiProblemBuilder>()
            .expect("Did not find DoubleHifiProblemBuilder in the params");

        let l_max = basis_recipe.l_max;
        let ordered_basis = self.basis_uncoupled(basis_recipe, hifi_problem);

        assert!(ordered_basis.is_sorted_by_key(|s| cast_variant!(s[0], SystemL).s));

        let angular_block_basis = (0..=l_max)
            .map(|l| {
                let red_basis = ordered_basis
                    .iter()
                    .filter(|s| matches!(s[0], SystemL(l_current) if l_current.s == l))
                    .cloned()
                    .collect::<StatesBasis<UncoupledAlkaliRotorAtomStates>>();

                (l, red_basis)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect::<Vec<(u32, StatesBasis<UncoupledAlkaliRotorAtomStates>)>>();

        let angular_blocks = angular_block_basis
            .iter()
            .map(|(l, basis)| {
                let n_centrifugal = operator_diagonal_mel!(&basis, |[n: RotorN]| {
                    rot_const * n.s.value() * (n.s.value() + 1.0)
                });

                let mut hifi = Mat::zeros(basis.len(), basis.len());
                if let Some(a_hifi) = hifi_problem.first.a_hifi {
                    hifi += operator_mel!(&basis, |[s: RotorS, i: RotorI]|
                        a_hifi * SpinOperators::dot(s, i)
                    )
                    .as_ref();
                }
                if let Some(a_hifi) = hifi_problem.second.a_hifi {
                    hifi += operator_mel!(&basis, |[s: AtomS, i: AtomI]|
                        a_hifi * SpinOperators::dot(s, i)
                    )
                    .as_ref();
                }

                let mut zeeman_prop =
                    operator_diagonal_mel!(&basis, |[s_rotor: RotorS, s_atom: AtomS]| {
                        -hifi_problem.first.gamma_e * s_rotor.ms.value()
                        -hifi_problem.second.gamma_e * s_atom.ms.value()
                    })
                    .into_backed();
                if let Some(gamma_i) = hifi_problem.first.gamma_i {
                    zeeman_prop += operator_diagonal_mel!(&basis, |[i: RotorI]|
                        -gamma_i * i.ms.value()
                    )
                    .as_ref()
                }
                if let Some(gamma_i) = hifi_problem.second.gamma_i {
                    zeeman_prop += operator_diagonal_mel!(&basis, |[i: AtomI]|
                        -gamma_i * i.ms.value()
                    )
                    .as_ref()
                }

                let spin_rot = operator_mel!(&basis, |[n: RotorN, s: RotorS]|
                    gamma_spin_rot * SpinOperators::dot(n, s)
                );

                let aniso_hifi = operator_mel!(&basis, |[n: RotorN, s: RotorS, i: RotorI]|
                    aniso_hifi * aniso_hifi_uncoupled_mel(n, s, i)
                );

                let field_inv = vec![
                    hifi,
                    aniso_hifi.into_backed(),
                    n_centrifugal.into_backed(),
                    spin_rot.into_backed(),
                ];

                let field_prop = vec![zeeman_prop];

                AngularBlock::new(AngMomentum(*l), field_inv, field_prop)
            })
            .collect();

        let singlet_potentials = self
            .singlet_potential
            .into_iter()
            .map(|(lambda, pot)| {
                let masking_singlet = operator_mel!(&ordered_basis,
                    |[l: SystemL, n: RotorN, s_rotor: RotorS, s_atom: AtomS]|
                        singlet_projection_uncoupled(s_rotor, s_atom)
                        * percival_coef_uncoupled_mel(lambda, n, l)
                );

                (pot, masking_singlet.into_backed())
            })
            .collect();

        let triplet_potentials = self
            .triplet_potential
            .into_iter()
            .map(|(lambda, pot)| {
                let masking_triplet = operator_mel!(&ordered_basis,
                    |[l: SystemL, n: RotorN,s_rotor: RotorS, s_atom: AtomS]|
                        triplet_projection_uncoupled(s_rotor, s_atom)
                        * percival_coef_uncoupled_mel(lambda, n, l)
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

    fn basis_uncoupled(
        &self,
        basis_recipe: &UncoupledRotorBasisRecipe,
        hifi_problem: &DoubleHifiProblemBuilder,
    ) -> StatesBasis<UncoupledAlkaliRotorAtomStates> {
        use UncoupledAlkaliRotorAtomStates::*;

        let l_max = basis_recipe.l_max;
        let n_max = basis_recipe.n_max;
        let parity = basis_recipe.parity;

        let l_system = (0..=l_max)
            .flat_map(|n_tot| into_variant(get_spin_basis(n_tot.into()), SystemL))
            .collect();

        let n_rotor = (0..=n_max)
            .flat_map(|n_tot| into_variant(get_spin_basis(n_tot.into()), RotorN))
            .collect();

        let s_rotor = into_variant(get_spin_basis(hifi_problem.first.s), RotorS);
        let s_atom = into_variant(get_spin_basis(hifi_problem.second.s), AtomS);
        let i_rotor = into_variant(get_spin_basis(hifi_problem.first.i), RotorI);
        let i_atom = into_variant(get_spin_basis(hifi_problem.second.i), AtomS);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(l_system))
            .push_state(StateBasis::new(n_rotor))
            .push_state(StateBasis::new(s_rotor))
            .push_state(StateBasis::new(s_atom))
            .push_state(StateBasis::new(i_rotor))
            .push_state(StateBasis::new(i_atom));

        let correct_parity = |l: Spin, n: Spin| match parity {
            ParityBlock::Positive => (l.s + n.s).double_value() % 4 == 0,
            ParityBlock::Negative => (l.s + n.s).double_value() % 4 != 0,
            ParityBlock::All => true,
        };

        let mut basis: StatesBasis<UncoupledAlkaliRotorAtomStates> =
            match hifi_problem.total_projection {
                Some(m_tot) => states
                    .iter_elements()
                    .filter(|b| {
                        let l = cast_variant!(b[0], SystemL);
                        let n = cast_variant!(b[1], RotorN);
                        let s_rotor = cast_variant!(b[2], RotorS);
                        let s_atom = cast_variant!(b[3], AtomS);
                        let i_rotor = cast_variant!(b[4], RotorI);
                        let i_atom = cast_variant!(b[5], AtomI);

                        l.ms + n.ms + s_rotor.ms + s_atom.ms + i_rotor.ms + i_atom.ms == m_tot
                            && correct_parity(l, n)
                    })
                    .collect(),
                None => states
                    .iter_elements()
                    .filter(|b| {
                        let l = cast_variant!(b[0], SystemL);
                        let n = cast_variant!(b[1], RotorN);

                        correct_parity(l, n)
                    })
                    .collect(),
            };

        basis.sort_by_key(|b| cast_variant!(b[0], SystemL).s);

        basis
    }
}
