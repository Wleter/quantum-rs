#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleSpin(pub u32, pub i32);

pub struct SpinOperators;

impl SpinOperators {
    pub fn proj_z(dspin_bra: DoubleSpin, dspin_ket: DoubleSpin) -> f64 {
        if dspin_bra == dspin_ket {
            dspin_bra.1 as f64 / 2.0
        } else {
            0.0
        }
    }

    pub fn ladder_plus(dspin_bra: DoubleSpin, dspin_ket: DoubleSpin) -> f64 {
        if dspin_bra.0 == dspin_ket.0 && dspin_bra.1 == dspin_ket.1 + 2 {
            (((dspin_ket.0 * (dspin_ket.0 + 2)) as i32 - dspin_bra.1 * dspin_ket.1) as f64).sqrt()
                / 2.0
        } else {
            0.0
        }
    }

    pub fn ladder_minus(dspin_bra: DoubleSpin, dspin_ket: DoubleSpin) -> f64 {
        if dspin_bra.0 == dspin_ket.0 && dspin_bra.1 + 2 == dspin_ket.1 {
            (((dspin_ket.0 * (dspin_ket.0 + 2)) as i32 - dspin_bra.1 * dspin_ket.1) as f64).sqrt()
                / 2.0
        } else {
            0.0
        }
    }

    pub fn dot(
        dspin1_braket: (DoubleSpin, DoubleSpin),
        dspin2_braket: (DoubleSpin, DoubleSpin),
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
    pub fn clebsch_gordan(dspin1: DoubleSpin, dspin2: DoubleSpin, dspin3: DoubleSpin) -> f64 {
        clebsch_gordan::clebsch_gordan(dspin1.0, dspin1.1, dspin2.0, dspin2.1, dspin3.0, dspin3.1)
    }
}

#[cfg(test)]
#[cfg(feature = "faer")]
mod test {
    use faer::{assert_matrix_eq, mat, Mat};

    use crate::{
        cast_variant,
        states::{operator::Operator, state::State, state_type::StateType, States},
    };

    use super::{DoubleSpin, SpinOperators};

    #[derive(Clone, Copy, PartialEq)]
    enum StateSep {
        Spin1(u32),
        Spin2(u32),
    }

    #[test]
    fn test_spin_ops() {
        let mut state = States::default();
        state
            .push_state(StateType::Irreducible(State::new(
                StateSep::Spin1(1),
                vec![-1, 1],
            )))
            .push_state(StateType::Irreducible(State::new(
                StateSep::Spin2(1),
                vec![-1, 1],
            )));

        let basis = state.get_basis();

        let op = Operator::<Mat<f64>>::from_mel(
            &basis,
            [StateSep::Spin1(0), StateSep::Spin2(0)],
            |[s1_braket, s2_braket]| {
                let s1_bra = cast_variant!(s1_braket.bra.0, StateSep::Spin1);
                let s1_bra = DoubleSpin(s1_bra, s1_braket.bra.1);

                let s2_bra = cast_variant!(s2_braket.bra.0, StateSep::Spin2);
                let s2_bra = DoubleSpin(s2_bra, s2_braket.bra.1);

                let s1_ket = cast_variant!(s1_braket.ket.0, StateSep::Spin1);
                let s1_ket = DoubleSpin(s1_ket, s1_braket.ket.1);

                let s2_ket = cast_variant!(s2_braket.ket.0, StateSep::Spin2);
                let s2_ket = DoubleSpin(s2_ket, s2_braket.ket.1);

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
        let singlet = State::new(0, vec![0]);
        let triplet = State::new(2, vec![-2, 0, 2]);
        combined.push_state(StateType::Sum(vec![singlet, triplet]));
        let basis_comb = combined.get_basis();

        let op = Operator::<Mat<f64>>::get_transformation(&basis, &basis_comb, |sep, comb| {
            let s1 = cast_variant!(sep.variants[0], StateSep::Spin1);
            let s1 = DoubleSpin(s1, sep.values[0]);

            let s2 = cast_variant!(sep.variants[1], StateSep::Spin2);
            let s2 = DoubleSpin(s2, sep.values[1]);

            let s = DoubleSpin(comb.variants[0], comb.values[0]);

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
