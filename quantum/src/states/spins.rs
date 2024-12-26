use clebsch_gordan::half_integer::{HalfI32, HalfU32};

pub fn spin_projections(s: HalfU32) -> Vec<HalfI32> {
    let ds = s.double_value() as i32;

    (-ds..=ds).step_by(2)
        .map(HalfI32::from_doubled)
        .collect()
}

pub fn sum_spin_projections(dspin1: HalfU32, dspin2: HalfU32) -> Vec<(HalfU32, Vec<HalfI32>)> {
    let dspin_max = (dspin1 + dspin2).double_value();
    let dspin_min = (dspin1.double_value() as i32 - dspin2.double_value() as i32).unsigned_abs();

    (dspin_min..=dspin_max)
        .step_by(2)
        .map(|s| {
            let s = HalfU32::from_doubled(s);
            (s, spin_projections(s))
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spin {
    pub s: HalfU32,
    pub ms: HalfI32
}

impl Spin {
    pub fn new(s: HalfU32, ms: HalfI32) -> Self {
        Self {
            s,
            ms,
        }
    }

    pub fn spin_type(&self) -> SpinType {
        if self.s.double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinType {
    Fermionic,
    Bosonic
}

pub struct SpinOperators;

impl SpinOperators {
    pub fn proj_z(spin_bra: Spin, spin_ket: Spin) -> f64 {
        if spin_bra == spin_ket {
            spin_bra.ms.value()
        } else {
            0.0
        }
    }

    pub fn ladder_plus(spin_bra: Spin, spin_ket: Spin) -> f64 {
        if spin_bra.s == spin_ket.s && spin_bra.ms.double_value() == spin_ket.ms.double_value() + 2 {
            (spin_ket.s.value() * (spin_ket.s.value() + 1.) - spin_bra.ms.value() * spin_ket.ms.value()).sqrt()
        } else {
            0.0
        }
    }

    pub fn ladder_minus(spin_bra: Spin, spin_ket: Spin) -> f64 {
        if spin_bra.s == spin_ket.s && spin_bra.ms.double_value() + 2 == spin_ket.ms.double_value() {
            (spin_ket.s.value() * (spin_ket.s.value() + 1.) - spin_bra.ms.value() * spin_ket.ms.value()).sqrt()
        } else {
            0.0
        }
    }

    pub fn dot(
        dspin1_braket: (Spin, Spin),
        dspin2_braket: (Spin, Spin),
    ) -> f64 {
        let val1 = Self::proj_z(dspin1_braket.0, dspin1_braket.1)
            * Self::proj_z(dspin2_braket.0, dspin2_braket.1);
        let val2 = 0.5 * Self::ladder_plus(dspin1_braket.0, dspin1_braket.1)
            * Self::ladder_minus(dspin2_braket.0, dspin2_braket.1);
        let val3 = 0.5 * Self::ladder_minus(dspin1_braket.0, dspin1_braket.1)
            * Self::ladder_plus(dspin2_braket.0, dspin2_braket.1);

        val1 + val2 + val3
    }

    /// Compute the Clebsch-Gordan coefficient <dspin1; dspin2 | dspin3>.
    pub fn clebsch_gordan(dspin1: Spin, dspin2: Spin, dspin3: Spin) -> f64 {
        clebsch_gordan::clebsch_gordan(dspin1.s, dspin1.ms, dspin2.s, dspin2.ms, dspin3.s, dspin3.ms)
    }
}

#[cfg(test)]
#[cfg(feature = "faer")]
mod test {
    use clebsch_gordan::{half_integer::HalfU32, half_u32};
    use faer::{assert_matrix_eq, mat, Mat};

    use crate::{
        cast_variant,
        states::{operator::Operator, spins::spin_projections, state::State, state_type::StateType, States},
    };

    use super::{Spin, SpinOperators};

    #[derive(Clone, Copy, PartialEq)]
    enum StateSep {
        Spin1(HalfU32),
        Spin2(HalfU32),
    }

    #[test]
    fn test_spin_ops() {
        let mut state = States::default();
        state
            .push_state(StateType::Irreducible(State::new(
                StateSep::Spin1(half_u32!(1/2)),
                spin_projections(half_u32!(1/2)),
            )))
            .push_state(StateType::Irreducible(State::new(
                StateSep::Spin2(half_u32!(1/2)),
                spin_projections(half_u32!(1/2)),
            )));

        let basis = state.get_basis();

        let op = Operator::<Mat<f64>>::from_mel(
            &basis,
            [StateSep::Spin1(half_u32!(0)), StateSep::Spin2(half_u32!(0))],
            |[s1_braket, s2_braket]| {
                let s1_bra = cast_variant!(s1_braket.bra.0, StateSep::Spin1);
                let s1_bra = Spin::new(s1_bra, s1_braket.bra.1);

                let s2_bra = cast_variant!(s2_braket.bra.0, StateSep::Spin2);
                let s2_bra = Spin::new(s2_bra, s2_braket.bra.1);

                let s1_ket = cast_variant!(s1_braket.ket.0, StateSep::Spin1);
                let s1_ket = Spin::new(s1_ket, s1_braket.ket.1);

                let s2_ket = cast_variant!(s2_braket.ket.0, StateSep::Spin2);
                let s2_ket = Spin::new(s2_ket, s2_braket.ket.1);

                SpinOperators::dot((s1_bra, s1_ket), (s2_bra, s2_ket))
            },
        );

        let expected = mat![
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ];
        assert_matrix_eq!(*op, expected);

        let mut combined = States::default();
        let singlet = State::new(half_u32!(0), spin_projections(half_u32!(0)));
        let triplet = State::new(half_u32!(1), spin_projections(half_u32!(1)));
        combined.push_state(StateType::Sum(vec![singlet, triplet]));
        let basis_comb = combined.get_basis();

        let op = Operator::<Mat<f64>>::get_transformation(&basis, &basis_comb, |sep, comb| {
            let s1 = cast_variant!(sep.variants[0], StateSep::Spin1);
            let s1 = Spin::new(s1, sep.values[0]);

            let s2 = cast_variant!(sep.variants[1], StateSep::Spin2);
            let s2 = Spin::new(s2, sep.values[1]);

            let s = Spin::new(comb.variants[0], comb.values[0]);

            SpinOperators::clebsch_gordan(s1, s2, s)
        });

        let expected = mat![
            [0.0, 0.7071, -0.7071, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.7071, 0.7071, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_matrix_eq!(*op, expected, comp = abs, tol = 1e-5);
    }
}
