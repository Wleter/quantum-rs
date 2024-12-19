use abm_states::{ABMStates, ABMStatesSep, ABMStatesValues, HifiStates, HifiStatesSep};
use consts::Consts;
use faer::{unzipped, zipped, Mat, MatRef};
use quantum::{
    cast_variant,
    states::{operator::Operator, spins::{spin_projections, sum_spin_projections}, state::State, state_type::StateType, States, StatesBasis},
    units::{energy_units::Energy, Au, Unit},
};
use utility::diagonalize;
use clebsch_gordan::{half_integer::{HalfI32, HalfU32}, half_u32};

pub mod abm_states;
pub mod consts;
pub mod utility;

#[derive(Clone, PartialEq)]
pub struct HifiProblemBuilder {
    pub s: HalfU32,
    pub i: HalfU32,
    pub gamma_e: f64,
    pub gamma_i: Option<f64>,
    pub a_hifi: Option<f64>,

    pub total_projection: Option<HalfI32>,
}

impl HifiProblemBuilder {
    pub fn new(double_s: HalfU32, double_i: HalfU32) -> Self {
        Self {
            s: double_s,
            i: double_i,
            gamma_e: -2.0 * Consts::BOHR_MAG,
            gamma_i: None,
            a_hifi: None,
            total_projection: None,
        }
    }

    pub fn with_nuclear_magneton(mut self, gamma_i: f64) -> Self {
        self.gamma_i = Some(gamma_i);

        self
    }

    pub fn with_custom_bohr_magneton(mut self, gamma_e: f64) -> Self {
        self.gamma_e = gamma_e;

        self
    }

    pub fn with_hyperfine_coupling(mut self, a_hifi: f64) -> Self {
        self.a_hifi = Some(a_hifi);

        self
    }

    pub fn with_total_projection(mut self, projection: HalfI32) -> Self {
        self.total_projection = Some(projection);

        self
    }

    pub fn build(self) -> ABMHifiProblem<HifiStates, HalfI32> {
        let mut states = States::default();
        let s = State::new(HifiStates::ElectronSpin(self.s), spin_projections(self.s));
        let i = State::new(HifiStates::NuclearSpin(self.i), spin_projections(self.i));

        states
            .push_state(StateType::Irreducible(s))
            .push_state(StateType::Irreducible(i));
        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|s| s.values.iter().copied().sum::<HalfI32>() == projection)
                .collect(),
            None => states.get_basis(),
        };

        let mut zeeman_prop =
            get_zeeman_prop!(basis, HifiStates::ElectronSpin, self.gamma_e).into_backed();
        if let Some(gamma_i) = self.gamma_i {
            zeeman_prop += get_zeeman_prop!(basis, HifiStates::NuclearSpin, gamma_i).as_ref();
        }

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.a_hifi {
            hifi += get_hifi!(
                basis,
                HifiStates::ElectronSpin,
                HifiStates::NuclearSpin,
                a_hifi
            )
            .as_ref()
        }

        ABMHifiProblem {
            basis,
            magnetic_inv: hifi,
            magnetic_prop: zeeman_prop,
        }
    }
}

#[derive(Clone)]
pub struct DoubleHifiProblemBuilder {
    pub first: HifiProblemBuilder,
    pub second: HifiProblemBuilder,

    pub total_projection: Option<HalfI32>,
    pub symmetry: Symmetry,
}

impl DoubleHifiProblemBuilder {
    pub fn new(first: HifiProblemBuilder, second: HifiProblemBuilder) -> Self {
        Self {
            first,
            second,
            symmetry: Symmetry::None,
            total_projection: None,
        }
    }

    pub fn new_homo(single: HifiProblemBuilder, symmetry: Symmetry) -> Self {
        Self {
            first: single.clone(),
            second: single,
            symmetry,
            total_projection: None,
        }
    }

    pub fn with_projection(mut self, projection: HalfI32) -> Self {
        self.total_projection = Some(projection);

        self
    }

    pub fn build(self) -> ABMHifiProblem<HifiStates, HalfI32> {
        let s1 = State::new(
            HifiStatesSep::ElSpin1(self.first.s),
            spin_projections(self.first.s),
        );
        let s2 = State::new(
            HifiStatesSep::ElSpin2(self.second.s),
            spin_projections(self.second.s),
        );
        let i1 = State::new(
            HifiStatesSep::NuclearSpin1(self.first.i),
            spin_projections(self.first.i),
        );
        let i2 = State::new(
            HifiStatesSep::NuclearSpin2(self.second.i),
            spin_projections(self.second.i),
        );

        let mut states = States::default();
        states
            .push_state(StateType::Irreducible(s1))
            .push_state(StateType::Irreducible(s2))
            .push_state(StateType::Irreducible(i1))
            .push_state(StateType::Irreducible(i2));

        let basis_sep = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|s| s.values.iter().copied().sum::<HalfI32>() == projection)
                .collect(),
            None => states.get_basis(),
        };

        let s_tot = sum_spin_projections(self.first.s, self.second.s)
            .into_iter()
            .map(|x| State::new(HifiStates::ElectronSpin(x.0), x.1))
            .collect();
        let i_tot = sum_spin_projections(self.first.i, self.second.i)
            .into_iter()
            .map(|x| State::new(HifiStates::NuclearSpin(x.0), x.1))
            .collect();

        let mut states = States::default();
        states
            .push_state(StateType::Sum(s_tot))
            .push_state(StateType::Sum(i_tot));

        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|s| s.values.iter().copied().sum::<HalfI32>() == projection)
                .collect(),
            None => states.get_basis(),
        };

        let mut zeeman_prop =
            get_zeeman_prop!(basis_sep, HifiStatesSep::ElSpin1, self.first.gamma_e).as_ref()
                + get_zeeman_prop!(basis_sep, HifiStatesSep::ElSpin2, self.second.gamma_e)
                    .as_ref();

        if let Some(gamma_i) = self.first.gamma_i {
            zeeman_prop +=
                get_zeeman_prop!(basis_sep, HifiStatesSep::NuclearSpin1, gamma_i).as_ref();
        }
        if let Some(gamma_i) = self.second.gamma_i {
            zeeman_prop +=
                get_zeeman_prop!(basis_sep, HifiStatesSep::NuclearSpin2, gamma_i).as_ref();
        }

        let mut hifi = Mat::zeros(basis_sep.len(), basis_sep.len());
        if let Some(a_hifi_1) = self.first.a_hifi {
            hifi += get_hifi!(
                basis_sep,
                HifiStatesSep::ElSpin1,
                HifiStatesSep::NuclearSpin1,
                a_hifi_1
            )
            .as_ref()
        }
        if let Some(a_hifi_2) = self.second.a_hifi {
            hifi += get_hifi!(
                basis_sep,
                HifiStatesSep::ElSpin2,
                HifiStatesSep::NuclearSpin2,
                a_hifi_2
            )
            .as_ref()
        }

        let s_max = self.first.s + self.second.s;
        let i_max = self.first.i + self.second.i;
        let basis = match self.symmetry {
            Symmetry::None => basis,
            Symmetry::Fermionic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s.variants[0], HifiStates::ElectronSpin);
                    let i_tot = cast_variant!(s.variants[1], HifiStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 != (s_tot + i_tot).double_value() % 4
                })
                .collect(),
            Symmetry::Bosonic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s.variants[0], HifiStates::ElectronSpin);
                    let i_tot = cast_variant!(s.variants[1], HifiStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 == (s_tot + i_tot).double_value() % 4
                })
                .collect(),
        };

        let transf = HifiStatesSep::convert_to_combined(&basis_sep, &basis);

        let zeeman_prop = transf.as_ref() * zeeman_prop * transf.transpose();
        let hifi = transf.as_ref() * hifi * transf.transpose();

        ABMHifiProblem {
            basis,
            magnetic_inv: hifi,
            magnetic_prop: zeeman_prop,
        }
    }
}

#[derive(Clone)]
pub struct ABMProblemBuilder {
    pub first: HifiProblemBuilder,
    pub second: HifiProblemBuilder,

    pub total_projection: Option<HalfI32>,
    pub symmetry: Symmetry,

    pub abm_vibrational: Option<ABMVibrational>,
}

impl ABMProblemBuilder {
    pub fn new(first: HifiProblemBuilder, second: HifiProblemBuilder) -> Self {
        Self {
            first,
            second,
            total_projection: None,
            symmetry: Symmetry::None,
            abm_vibrational: None,
        }
    }

    pub fn new_homo(single: HifiProblemBuilder, symmetry: Symmetry) -> Self {
        Self {
            first: single.clone(),
            second: single,
            total_projection: None,
            symmetry,
            abm_vibrational: None,
        }
    }

    pub fn with_vibrational(mut self, vibrational: ABMVibrational) -> Self {
        self.abm_vibrational = Some(vibrational);

        self
    }

    pub fn with_projection(mut self, projection: HalfI32) -> Self {
        self.total_projection = Some(projection);

        self
    }

    pub fn build(self) -> ABMHifiProblem<ABMStates, ABMStatesValues> {
        assert!(
            self.first.s == half_u32!(1/2) && self.second.s == half_u32!(1/2),
            "ABM problem is done under the assumption of s = 1/2"
        );

        let mut states = States::default();
        let s1 = State::new(
            ABMStatesSep::ElSpin1(self.first.s),
            spin_projections(self.first.s)
                .into_iter()
                .map(|x| ABMStatesValues::Proj(x))
                .collect(),
        );
        let s2 = State::new(
            ABMStatesSep::ElSpin2(self.second.s),
            spin_projections(self.second.s)
                .into_iter()
                .map(|x| ABMStatesValues::Proj(x))
                .collect(),
        );
        let i1 = State::new(
            ABMStatesSep::NuclearSpin1(self.first.i),
            spin_projections(self.first.i)
                .into_iter()
                .map(|x| ABMStatesValues::Proj(x))
                .collect(),
        );
        let i2 = State::new(
            ABMStatesSep::NuclearSpin2(self.second.i),
            spin_projections(self.second.i)
                .into_iter()
                .map(|x| ABMStatesValues::Proj(x))
                .collect(),
        );

        assert!(
            self.abm_vibrational.is_some(),
            "vibrational states not provided by function with_vibrational"
        );
        let abm_vibrational = self.abm_vibrational.unwrap();
        let vib_values: Vec<ABMStatesValues> = abm_vibrational.states()
            .into_iter()
            .map(|x| ABMStatesValues::Vib(x))
            .collect();

        let bounds_sep = State::new(ABMStatesSep::Vibrational, vib_values.clone());

        states
            .push_state(StateType::Irreducible(s1))
            .push_state(StateType::Irreducible(s2))
            .push_state(StateType::Irreducible(i1))
            .push_state(StateType::Irreducible(i2))
            .push_state(StateType::Irreducible(bounds_sep));

        let basis_sep = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|s| {
                    s.values.iter()
                        .filter(|e| matches!(e, ABMStatesValues::Proj(_)))
                        .map(|e| *cast_variant!(e, ABMStatesValues::Proj))
                        .sum::<HalfI32>() == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let s_tot = sum_spin_projections(self.first.s, self.second.s)
            .into_iter()
            .map(|x| {
                let projections = x.1.into_iter().map(|p| ABMStatesValues::Proj(p)).collect();
                State::new(ABMStates::ElectronSpin(x.0), projections)
            })
            .collect();
        let i_tot = sum_spin_projections(self.first.i, self.second.i)
            .into_iter()
            .map(|x| {
                let projections = x.1.into_iter().map(|p| ABMStatesValues::Proj(p)).collect();
                State::new(ABMStates::NuclearSpin(x.0), projections)
            })
            .collect();
        let bounds = State::new(ABMStates::Vibrational, vib_values);

        let mut states = States::default();
        states
            .push_state(StateType::Sum(s_tot))
            .push_state(StateType::Sum(i_tot))
            .push_state(StateType::Irreducible(bounds));

        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|s| {
                    s.values.iter()
                        .filter(|e| matches!(e, ABMStatesValues::Proj(_)))
                        .map(|e| *cast_variant!(e, ABMStatesValues::Proj))
                        .sum::<HalfI32>() == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let s_max = self.first.s + self.second.s;
        let i_max = self.first.i + self.second.i;
        let basis = match self.symmetry {
            Symmetry::None => basis,
            Symmetry::Fermionic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s.variants[0], ABMStates::ElectronSpin);
                    let i_tot = cast_variant!(s.variants[1], ABMStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 != (s_tot + i_tot).double_value() % 4
                })
                .collect(),
            Symmetry::Bosonic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s.variants[0], ABMStates::ElectronSpin);
                    let i_tot = cast_variant!(s.variants[1], ABMStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 == (s_tot + i_tot).double_value() % 4
                })
                .collect(),
        };

        let mut zeeman_prop =
            get_zeeman_prop!(basis_sep, ABMStatesSep::ElSpin1, self.first.gamma_e; ABMStatesValues::Proj).as_ref()
                + get_zeeman_prop!(basis_sep, ABMStatesSep::ElSpin2, self.second.gamma_e; ABMStatesValues::Proj).as_ref();

        if let Some(gamma_i) = self.first.gamma_i {
            zeeman_prop +=
                get_zeeman_prop!(basis_sep, ABMStatesSep::NuclearSpin1, gamma_i; ABMStatesValues::Proj).as_ref();
        }
        if let Some(gamma_i) = self.second.gamma_i {
            zeeman_prop +=
                get_zeeman_prop!(basis_sep, ABMStatesSep::NuclearSpin2, gamma_i; ABMStatesValues::Proj).as_ref();
        }

        let mut hifi = Mat::zeros(basis_sep.len(), basis_sep.len());
        if let Some(a_hifi_1) = self.first.a_hifi {
            hifi += get_hifi!(basis_sep, ABMStatesSep::ElSpin1, ABMStatesSep::NuclearSpin1,
                            a_hifi_1, with ABMStatesSep::Vibrational; ABMStatesValues::Proj)
            .as_ref()
        }
        if let Some(a_hifi_2) = self.second.a_hifi {
            hifi += get_hifi!(basis_sep, ABMStatesSep::ElSpin2, ABMStatesSep::NuclearSpin2,
                            a_hifi_2, with ABMStatesSep::Vibrational; ABMStatesValues::Proj)
            .as_ref()
        }

        let transf = ABMStatesSep::convert_to_combined(&basis_sep, &basis);

        let bound_states = Operator::from_diagonal_mel(
            &basis,
            [ABMStates::ElectronSpin(half_u32!(0)), ABMStates::Vibrational],
            |[s_ket, vib_ket]| {
                let s = cast_variant!(s_ket.0, ABMStates::ElectronSpin);
                let vib = cast_variant!(vib_ket.1, ABMStatesValues::Vib);

                if s == half_u32!(1) {
                    abm_vibrational.triplet_state(vib)
                } else {
                    abm_vibrational.singlet_state(vib)
                }
            },
        );

        let fc_factors = Operator::from_mel(
            &basis,
            [
                ABMStates::ElectronSpin(half_u32!(0)),
                ABMStates::NuclearSpin(half_u32!(0)),
                ABMStates::Vibrational,
            ],
            |[s_braket, _, vib_braket]| {
                let s_ket = cast_variant!(s_braket.ket.0, ABMStates::ElectronSpin);
                let s_bra = cast_variant!(s_braket.bra.0, ABMStates::ElectronSpin);
                let vib_bra = cast_variant!(vib_braket.bra.1, ABMStatesValues::Vib);
                let vib_ket = cast_variant!(vib_braket.ket.1, ABMStatesValues::Vib);

                if s_bra == s_ket {
                    if vib_bra == vib_ket {
                        1.
                    } else {
                        0.
                    }
                } else if s_bra == half_u32!(1) {
                    assert!(s_ket == half_u32!(0));
                    abm_vibrational.fc_factor(vib_ket, vib_bra)
                } else {
                    assert!(s_bra == half_u32!(0));
                    assert!(s_ket == half_u32!(1));
                    abm_vibrational.fc_factor(vib_bra, vib_ket)
                }
            },
        );

        let mut hifi = &*transf * hifi * transf.transpose();
        zipped!(hifi.as_mut(), fc_factors.as_ref())
            .for_each(|unzipped!(mut h, fc)| h.write(h.read() * fc.read()));

        let zeeman_prop = &*transf * zeeman_prop * transf.transpose();

        ABMHifiProblem {
            basis,
            magnetic_inv: hifi + bound_states.as_ref(),
            magnetic_prop: zeeman_prop,
        }
    }
}

#[derive(Clone)]
pub enum Symmetry {
    None,
    Fermionic,
    Bosonic,
}

#[derive(Clone)]
pub struct ABMVibrational {
    singlet_states: Vec<f64>,
    triplet_states: Vec<f64>,
    fc_factors: Mat<f64>,
}

impl ABMVibrational {
    /// Creates struct for storing vibrational abm states
    /// and Franck-Condon factors.
    ///
    /// Franck-Condon factors are expected to be matrix of elements
    /// <S = 0, v_S | S' = 1, v_S'>
    pub fn new<U: Unit>(
        singlet_states: Vec<Energy<U>>,
        triplet_states: Vec<Energy<U>>,
        fc_factors: Mat<f64>,
    ) -> Self {
        let singlet_states: Vec<f64> = singlet_states.into_iter().map(|x| x.to_au()).collect();
        let triplet_states: Vec<f64> = triplet_states.into_iter().map(|x| x.to_au()).collect();

        assert!(singlet_states.len() == triplet_states.len(), "non-compatible states sizes");
        assert!(singlet_states.len().pow(2) == fc_factors.nrows() * fc_factors.ncols()
            && fc_factors.nrows() == fc_factors.ncols(),
            "wrong dimensions of franck-condon factors");

        assert!(
            singlet_states.windows(2).all(|w| w[0] > w[1]),
            "states should be sorted from -1 state"
        );
        assert!(
            triplet_states.windows(2).all(|w| w[0] > w[1]),
            "states should be sorted from -1 state"
        );

        Self {
            singlet_states,
            triplet_states,
            fc_factors,
        }
    }

    pub fn states_no(&self) -> usize {
        self.singlet_states.len()
    }

    pub fn states(&self) -> Vec<i32> {
        (-(self.states_no() as i32)..0).rev().collect()
    }

    pub fn triplet_state(&self, vib: i32) -> f64 {
        self.triplet_states[(-vib - 1) as usize]
    }

    pub fn singlet_state(&self, vib: i32) -> f64 {
        self.singlet_states[(-vib - 1) as usize]
    }

    pub fn fc_factor(&self, vib_singlet: i32, vib_triplet: i32) -> f64 {
        let i = (-vib_singlet - 1) as usize;
        let j = (-vib_triplet - 1) as usize;

        self.fc_factors[(i, j)]
    }
}

pub struct ABMHifiProblem<T, V> {
    basis: StatesBasis<T, V>,
    magnetic_inv: Mat<f64>,
    magnetic_prop: Mat<f64>,
}

impl<T, V> ABMHifiProblem<T, V> {
    pub fn new_custom(
        basis: StatesBasis<T, V>,
        magnetic_inv: Mat<f64>,
        magnetic_prop: Mat<f64>,
    ) -> Self {
        Self {
            basis,
            magnetic_inv,
            magnetic_prop,
        }
    }

    pub fn states_at(&self, magnetic_field: f64) -> (Vec<Energy<Au>>, Mat<f64>) {
        let hamiltonian = &self.magnetic_inv + magnetic_field * &self.magnetic_prop;

        let (energies, vectors) = diagonalize(hamiltonian.as_ref());

        let energies = energies.into_iter().map(|x| Energy(x, Au)).collect();

        (energies, vectors)
    }

    pub fn get_magnetic_inv(&self) -> MatRef<f64> {
        self.magnetic_inv.as_ref()
    }

    pub fn get_magnetic_prop(&self) -> MatRef<f64> {
        self.magnetic_prop.as_ref()
    }

    pub fn get_basis(&self) -> &StatesBasis<T, V> {
        &self.basis
    }

    pub fn size(&self) -> (usize, usize) {
        (self.magnetic_prop.nrows(), self.magnetic_prop.ncols())
    }
}
