use abm_states::{ABMStates, ABMStatesSep, HifiStates, HifiStatesSep};
use clebsch_gordan::{
    half_integer::{HalfI32, HalfU32},
    hu32,
};
use consts::Consts;
use faer::{Mat, MatRef, unzipped, zipped};
use quantum::{
    cast_variant, operator_diagonal_mel, operator_mel,
    states::{
        States, StatesBasis,
        braket::kron_delta,
        operator::Operator,
        spins::{SpinOperators, get_spin_basis, get_summed_spin_basis},
        state::{StateBasis, into_variant},
    },
    units::{
        Au,
        energy_units::{Energy, EnergyUnit},
    },
};
use utility::diagonalize;

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
            gamma_e: -2.00231930436256 * Consts::BOHR_MAG,
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

    pub fn build(self) -> ABMHifiProblem<HifiStates> {
        let mut states = States::default();
        let s = into_variant(get_spin_basis(self.s), HifiStates::ElectronSpin);
        let i = into_variant(get_spin_basis(self.i), HifiStates::NuclearSpin);

        states
            .push_state(StateBasis::new(s))
            .push_state(StateBasis::new(i));

        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|b| {
                    let s = cast_variant!(b[0], HifiStates::ElectronSpin);
                    let i = cast_variant!(b[1], HifiStates::NuclearSpin);

                    s.ms + i.ms == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let mut zeeman_prop = operator_diagonal_mel!(&basis, |[s: HifiStates::ElectronSpin]| {
            -self.gamma_e * s.ms.value()
        })
        .into_backed();

        if let Some(gamma_i) = self.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis, |[i: HifiStates::NuclearSpin]| {
                -gamma_i * i.ms.value()
            })
            .as_ref();
        }

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.a_hifi {
            hifi +=
                operator_mel!(&basis, |[s: HifiStates::ElectronSpin, i: HifiStates::NuclearSpin]| {
                    a_hifi * SpinOperators::dot(s, i)
                })
                .as_ref();
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

    pub fn build(self) -> ABMHifiProblem<HifiStates> {
        let s1 = into_variant(get_spin_basis(self.first.s), HifiStatesSep::ElSpin1);
        let s2 = into_variant(get_spin_basis(self.second.s), HifiStatesSep::ElSpin2);
        let i1 = into_variant(get_spin_basis(self.first.i), HifiStatesSep::NuclearSpin1);
        let i2 = into_variant(get_spin_basis(self.second.i), HifiStatesSep::NuclearSpin2);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(s1))
            .push_state(StateBasis::new(s2))
            .push_state(StateBasis::new(i1))
            .push_state(StateBasis::new(i2));

        let basis_sep = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|b| {
                    let s1 = cast_variant!(b[0], HifiStatesSep::ElSpin1);
                    let s2 = cast_variant!(b[1], HifiStatesSep::ElSpin2);
                    let i1 = cast_variant!(b[2], HifiStatesSep::NuclearSpin1);
                    let i2 = cast_variant!(b[3], HifiStatesSep::NuclearSpin2);

                    s1.ms + s2.ms + i1.ms + i2.ms == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let s_tot = into_variant(
            get_summed_spin_basis(self.first.s, self.second.s),
            HifiStates::ElectronSpin,
        );
        let i_tot = into_variant(
            get_summed_spin_basis(self.first.i, self.second.i),
            HifiStates::NuclearSpin,
        );

        let mut states = States::default();
        states
            .push_state(StateBasis::new(s_tot))
            .push_state(StateBasis::new(i_tot));

        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|b| {
                    let s_tot = cast_variant!(b[0], HifiStates::ElectronSpin);
                    let i_tot = cast_variant!(b[1], HifiStates::NuclearSpin);

                    s_tot.ms + i_tot.ms == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let mut zeeman_prop = operator_diagonal_mel!(&basis_sep,
            |[s1: HifiStatesSep::ElSpin1, s2: HifiStatesSep::ElSpin2]| {
                -self.first.gamma_e * s1.ms.value() - self.second.gamma_e * s2.ms.value()
            }
        )
        .into_backed();

        if let Some(gamma_i) = self.first.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis_sep, |[i: HifiStatesSep::NuclearSpin1]| {
                -gamma_i * i.ms.value()
            })
            .as_ref()
        }
        if let Some(gamma_i) = self.second.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis_sep, |[i: HifiStatesSep::NuclearSpin2]| {
                -gamma_i * i.ms.value()
            })
            .as_ref()
        }

        let mut hifi = Mat::zeros(basis_sep.len(), basis_sep.len());
        if let Some(a_hifi) = self.first.a_hifi {
            hifi += operator_mel!(&basis_sep,
                |[s: HifiStatesSep::ElSpin1, i: HifiStatesSep::NuclearSpin1]| {
                    a_hifi * SpinOperators::dot(s, i)
                }
            )
            .as_ref()
        }
        if let Some(a_hifi) = self.second.a_hifi {
            hifi += operator_mel!(&basis_sep,
                |[s: HifiStatesSep::ElSpin2, i: HifiStatesSep::NuclearSpin2]| {
                    a_hifi * SpinOperators::dot(s, i)
                }
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
                    let s_tot = cast_variant!(s[0], HifiStates::ElectronSpin);
                    let i_tot = cast_variant!(s[1], HifiStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 != (s_tot.s + i_tot.s).double_value() % 4
                })
                .collect(),
            Symmetry::Bosonic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s[0], HifiStates::ElectronSpin);
                    let i_tot = cast_variant!(s[1], HifiStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 == (s_tot.s + i_tot.s).double_value() % 4
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

    pub fn build(self) -> ABMHifiProblem<ABMStates> {
        assert!(
            self.first.s == hu32!(1 / 2) && self.second.s == hu32!(1 / 2),
            "ABM problem is done under the assumption of s = 1/2"
        );

        assert!(
            self.abm_vibrational.is_some(),
            "vibrational states not provided by function with_vibrational"
        );
        let abm_vibrational = self.abm_vibrational.unwrap();

        let s1 = into_variant(get_spin_basis(self.first.s), ABMStatesSep::ElSpin1);
        let s2 = into_variant(get_spin_basis(self.second.s), ABMStatesSep::ElSpin2);
        let i1 = into_variant(get_spin_basis(self.first.i), ABMStatesSep::NuclearSpin1);
        let i2 = into_variant(get_spin_basis(self.second.i), ABMStatesSep::NuclearSpin2);
        let bounds_sep = into_variant(abm_vibrational.states(), ABMStatesSep::Vibrational);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(s1))
            .push_state(StateBasis::new(s2))
            .push_state(StateBasis::new(i1))
            .push_state(StateBasis::new(i2))
            .push_state(StateBasis::new(bounds_sep));

        let basis_sep = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|b| {
                    let s1 = cast_variant!(b[0], ABMStatesSep::ElSpin1);
                    let s2 = cast_variant!(b[1], ABMStatesSep::ElSpin2);
                    let i1 = cast_variant!(b[2], ABMStatesSep::NuclearSpin1);
                    let i2 = cast_variant!(b[3], ABMStatesSep::NuclearSpin2);

                    s1.ms + s2.ms + i1.ms + i2.ms == projection
                })
                .collect(),
            None => states.get_basis(),
        };

        let s_tot = StateBasis::new(into_variant(
            get_summed_spin_basis(self.first.s, self.second.s),
            ABMStates::ElectronSpin,
        ));
        let i_tot = StateBasis::new(into_variant(
            get_summed_spin_basis(self.first.i, self.second.i),
            ABMStates::NuclearSpin,
        ));
        let bounds = StateBasis::new(into_variant(
            abm_vibrational.states(),
            ABMStates::Vibrational,
        ));

        let mut states = States::default();
        states
            .push_state(s_tot)
            .push_state(i_tot)
            .push_state(bounds);

        let basis = match self.total_projection {
            Some(projection) => states
                .iter_elements()
                .filter(|b| {
                    let s_tot = cast_variant!(b[0], ABMStates::ElectronSpin);
                    let i_tot = cast_variant!(b[1], ABMStates::NuclearSpin);

                    s_tot.ms + i_tot.ms == projection
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
                    let s_tot = cast_variant!(s[0], ABMStates::ElectronSpin);
                    let i_tot = cast_variant!(s[1], ABMStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 != (s_tot.s + i_tot.s).double_value() % 4
                })
                .collect(),
            Symmetry::Bosonic => basis
                .into_iter()
                .filter(|s| {
                    let s_tot = cast_variant!(s[0], ABMStates::ElectronSpin);
                    let i_tot = cast_variant!(s[1], ABMStates::NuclearSpin);

                    (s_max + i_max).double_value() % 4 == (s_tot.s + i_tot.s).double_value() % 4
                })
                .collect(),
        };

        let mut zeeman_prop = operator_diagonal_mel!(&basis_sep,
            |[s1: ABMStatesSep::ElSpin1, s2: ABMStatesSep::ElSpin2]| {
                -self.first.gamma_e * s1.ms.value() - self.second.gamma_e * s2.ms.value()
            }
        )
        .into_backed();

        if let Some(gamma_i) = self.first.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis_sep, |[i: ABMStatesSep::NuclearSpin1]| {
                -gamma_i * i.ms.value()
            })
            .as_ref()
        }
        if let Some(gamma_i) = self.second.gamma_i {
            zeeman_prop += operator_diagonal_mel!(&basis_sep, |[i: ABMStatesSep::NuclearSpin2]| {
                -gamma_i * i.ms.value()
            })
            .as_ref()
        }

        let mut hifi = Mat::zeros(basis_sep.len(), basis_sep.len());
        if let Some(a_hifi) = self.first.a_hifi {
            hifi += operator_mel!(&basis_sep,
                |[s: ABMStatesSep::ElSpin1, i: ABMStatesSep::NuclearSpin1, _v: ABMStatesSep::Vibrational]| {
                    a_hifi * SpinOperators::dot(s, i)
                }
            ).as_ref()
        }
        if let Some(a_hifi) = self.second.a_hifi {
            hifi += operator_mel!(&basis_sep,
                |[s: ABMStatesSep::ElSpin2, i: ABMStatesSep::NuclearSpin2, _v: ABMStatesSep::Vibrational]| {
                    a_hifi * SpinOperators::dot(s, i)
                }
            ).as_ref()
        }

        let transf = ABMStatesSep::convert_to_combined(&basis_sep, &basis);

        let bound_states = operator_diagonal_mel!(&basis,
            |[
                s: ABMStates::ElectronSpin,
                vib: ABMStates::Vibrational
            ]| {
            if s.s == hu32!(1) {
                abm_vibrational.triplet_state(vib)
            } else {
                abm_vibrational.singlet_state(vib)
            }
        });

        let fc_factors = operator_mel!(&basis,
            |[
                s: ABMStates::ElectronSpin,
                _i: ABMStates::NuclearSpin,
                vib: ABMStates::Vibrational
            ]| {
                if s.bra.s == s.ket.s {
                    kron_delta([vib])
                } else if s.bra.s == hu32!(1) {
                    assert!(s.ket.s == hu32!(0));
                    abm_vibrational.fc_factor(vib.ket, vib.bra)
                } else {
                    assert!(s.bra.s == hu32!(0));
                    assert!(s.ket.s == hu32!(1));
                    abm_vibrational.fc_factor(vib.bra, vib.ket)
                }
            }
        );

        let mut hifi = &*transf * hifi * transf.transpose();
        zipped!(hifi.as_mut(), fc_factors.as_ref()).for_each(|unzipped!(h, fc)| *h *= fc);

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
    pub fn new<U: EnergyUnit>(
        singlet_states: Vec<Energy<U>>,
        triplet_states: Vec<Energy<U>>,
        fc_factors: Mat<f64>,
    ) -> Self {
        let singlet_states: Vec<f64> = singlet_states.into_iter().map(|x| x.to_au()).collect();
        let triplet_states: Vec<f64> = triplet_states.into_iter().map(|x| x.to_au()).collect();

        assert!(
            singlet_states.len() == triplet_states.len(),
            "non-compatible states sizes"
        );
        assert!(
            singlet_states.len().pow(2) == fc_factors.nrows() * fc_factors.ncols()
                && fc_factors.nrows() == fc_factors.ncols(),
            "wrong dimensions of franck-condon factors"
        );

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

pub struct ABMHifiProblem<T> {
    basis: StatesBasis<T>,
    magnetic_inv: Mat<f64>,
    magnetic_prop: Mat<f64>,
}

impl<T> ABMHifiProblem<T> {
    pub fn new_custom(
        basis: StatesBasis<T>,
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

    pub fn get_basis(&self) -> &StatesBasis<T> {
        &self.basis
    }

    pub fn size(&self) -> (usize, usize) {
        (self.magnetic_prop.nrows(), self.magnetic_prop.ncols())
    }
}
