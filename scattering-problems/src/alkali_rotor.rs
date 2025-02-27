use abm::{HifiProblemBuilder, utility::diagonalize};
use clebsch_gordan::hu32;
use faer::Mat;
use quantum::{
    cast_variant, operator_diagonal_mel, operator_mel,
    params::{Params, particle_factory::RotConst},
    states::{
        States, StatesBasis,
        operator::Operator,
        spins::{Spin, SpinOperators, get_spin_basis},
        state::{StateBasis, into_variant},
    },
};

use crate::{
    alkali_rotor_atom::TramBasisRecipe,
    utility::{
        AngularPair, AnisoHifi, GammaSpinRot, aniso_hifi_tram_mel, aniso_hifi_uncoupled_mel,
        create_angular_pairs, spin_rot_tram_mel,
    },
};

#[derive(Clone)]
pub struct AlkaliRotorProblemBuilder {
    hifi_problem: HifiProblemBuilder,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlkaliRotorStates {
    /// (l, j, j_tot)
    Angular(AngularPair),
    NTot(Spin),
    RotorS(Spin),
    RotorI(Spin),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UncoupledAlkaliRotorStates {
    RotorN(Spin),
    RotorS(Spin),
    RotorI(Spin),
}

impl AlkaliRotorProblemBuilder {
    pub fn new(hifi_problem: HifiProblemBuilder) -> Self {
        assert!(hifi_problem.s == hu32!(1 / 2));

        Self { hifi_problem }
    }

    pub fn build(self, params: &Params, basis_recipe: &TramBasisRecipe) -> AlkaliRotorProblem {
        use AlkaliRotorStates::{Angular, NTot, RotorI, RotorS};

        let rot_const = params
            .get::<RotConst>()
            .expect("Did not find RotConst parameter in the params")
            .0;
        let gamma_spin_rot = params.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = params.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;

        let l_max = basis_recipe.l_max;
        let n_max = basis_recipe.n_max;
        let n_tot_max = basis_recipe.n_tot_max;
        let parity = basis_recipe.parity;

        let angular_states = create_angular_pairs(l_max, n_max, n_tot_max, parity);
        let angular_states = into_variant(angular_states, Angular);

        let total_angular = (0..=n_tot_max)
            .map(|n_tot| into_variant(get_spin_basis(n_tot.into()), NTot))
            .flatten()
            .collect();

        let s_rotor = into_variant(get_spin_basis(self.hifi_problem.s), RotorS);
        let i_rotor = into_variant(get_spin_basis(self.hifi_problem.i), RotorI);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(angular_states))
            .push_state(StateBasis::new(total_angular))
            .push_state(StateBasis::new(s_rotor))
            .push_state(StateBasis::new(i_rotor));

        let basis = match self.hifi_problem.total_projection {
            Some(m_tot) => states
                .iter_elements()
                .filter(|b| {
                    let ang = cast_variant!(b[0], Angular);
                    let m_n_tot = cast_variant!(b[1], NTot);
                    let s_rotor = cast_variant!(b[2], RotorS);
                    let i_rotor = cast_variant!(b[3], RotorI);

                    m_n_tot.ms + s_rotor.ms + i_rotor.ms == m_tot
                        && (ang.l + ang.n) >= m_n_tot.s
                        && (ang.l + m_n_tot.s) >= ang.n
                        && (ang.n + m_n_tot.s) >= ang.l
                })
                .collect(),
            None => states
                .iter_elements()
                .filter(|b| {
                    let ang = cast_variant!(b[0], Angular);
                    let m_n_tot = cast_variant!(b[1], NTot);

                    (ang.l + ang.n) >= m_n_tot.s
                        && (ang.l + m_n_tot.s) >= ang.n
                        && (ang.n + m_n_tot.s) >= ang.l
                })
                .collect(),
        };

        let n_centrifugal = operator_diagonal_mel!(&basis, |[ang: Angular]|
            rot_const * ang.n.value() * (ang.n.value() + 1.0)
        );

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.hifi_problem.a_hifi {
            hifi += operator_mel!(&basis, |[s: RotorS, i: RotorI]| {
                a_hifi * SpinOperators::dot(s, i)
            })
            .as_ref();
        }

        let mut zeeman_prop = operator_diagonal_mel!(&basis, |[s_rotor: RotorS]|
            -self.hifi_problem.gamma_e * s_rotor.ms.value()
        )
        .into_backed();
        if let Some(gamma_i) = self.hifi_problem.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis, |[i: RotorI]|
                -gamma_i * i.ms.value()
            )
            .as_ref()
        }

        let spin_rot = operator_mel!(&basis, |[ang: Angular, n_tot: NTot, s: RotorS]|
            gamma_spin_rot * spin_rot_tram_mel(ang, n_tot, s)
        );

        let aniso_hifi = operator_mel!(&basis, |[ang: Angular, n_tot: NTot, s: RotorS, i: RotorI]|
            aniso_hifi * aniso_hifi_tram_mel(ang, n_tot, s, i)
        );

        AlkaliRotorProblem {
            basis,
            mag_inv: hifi + aniso_hifi.as_ref() + n_centrifugal.as_ref() + spin_rot.as_ref(),
            mag_prop: zeeman_prop,
        }
    }

    pub fn build_uncoupled(self, params: &Params, n_max: u32) -> UncoupledAlkaliRotorProblem {
        use UncoupledAlkaliRotorStates::{RotorI, RotorN, RotorS};

        let rot_const = params
            .get::<RotConst>()
            .expect("Did not find RotConst parameter in the params")
            .0;
        let gamma_spin_rot = params.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = params.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;

        let n_rotor = (0..=n_max)
            .map(|n_tot| into_variant(get_spin_basis(n_tot.into()), RotorN))
            .flatten()
            .collect();

        let s_rotor = into_variant(get_spin_basis(self.hifi_problem.s), RotorS);
        let i_rotor = into_variant(get_spin_basis(self.hifi_problem.i), RotorI);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(n_rotor))
            .push_state(StateBasis::new(s_rotor))
            .push_state(StateBasis::new(i_rotor));

        let basis = match self.hifi_problem.total_projection {
            Some(m_tot) => states
                .iter_elements()
                .filter(|b| {
                    let n_rotor = cast_variant!(b[0], RotorN);
                    let s_rotor = cast_variant!(b[1], RotorS);
                    let i_rotor = cast_variant!(b[2], RotorI);

                    n_rotor.ms + s_rotor.ms + i_rotor.ms == m_tot
                })
                .collect(),
            None => states.iter_elements().collect(),
        };

        let n_centrifugal = operator_diagonal_mel!(&basis, |[n: RotorN]|
            rot_const * n.s.value() * (n.s.value() + 1.)
        );

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.hifi_problem.a_hifi {
            hifi += operator_mel!(&basis, |[s: RotorS, i: RotorI]|
                a_hifi * SpinOperators::dot(s, i)
            )
            .as_ref();
        }

        let mut zeeman_prop = operator_diagonal_mel!(&basis, |[s_rotor: RotorS]|
            -self.hifi_problem.gamma_e * s_rotor.ms.value()
        )
        .into_backed();
        if let Some(gamma_i) = self.hifi_problem.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis, |[i: RotorI]|
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

        UncoupledAlkaliRotorProblem {
            basis,
            mag_inv: hifi + aniso_hifi.as_ref() + n_centrifugal.as_ref() + spin_rot.as_ref(),
            mag_prop: zeeman_prop,
        }
    }
}

pub struct AlkaliRotorProblem {
    pub basis: StatesBasis<AlkaliRotorStates>,
    mag_inv: Mat<f64>,
    mag_prop: Mat<f64>,
}

impl AlkaliRotorProblem {
    pub fn levels(&self, field: f64) -> (Vec<f64>, Mat<f64>) {
        let internal = &self.mag_inv + field * &self.mag_prop;

        diagonalize(internal.as_ref())
    }
}

pub struct UncoupledAlkaliRotorProblem {
    pub basis: StatesBasis<UncoupledAlkaliRotorStates>,
    mag_inv: Mat<f64>,
    mag_prop: Mat<f64>,
}

impl UncoupledAlkaliRotorProblem {
    pub fn levels(&self, field: f64) -> (Vec<f64>, Mat<f64>) {
        let internal = &self.mag_inv + field * &self.mag_prop;

        diagonalize(internal.as_ref())
    }
}
