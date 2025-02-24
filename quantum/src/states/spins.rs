use clebsch_gordan::half_integer::{HalfI32, HalfU32};

use super::braket::Braket;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Spin {
    pub s: HalfU32,
    pub ms: HalfI32,
}

impl Spin {
    pub fn new(s: HalfU32, ms: HalfI32) -> Self {
        Self { s, ms }
    }

    pub fn spin_type(&self) -> SpinType {
        if self.s.double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }
}

pub fn get_spin_basis(s: HalfU32) -> Vec<Spin> {
    let ds = s.double_value() as i32;

    (-ds..=ds).step_by(2)
        .map(|dms| Spin::new(s, HalfI32::from_doubled(dms)))
        .collect()
}

pub fn get_summed_spin_basis(dspin1: HalfU32, dspin2: HalfU32) -> Vec<Spin> {
    let dspin_max = (dspin1 + dspin2).double_value();
    let dspin_min = (dspin1.double_value() as i32 - dspin2.double_value() as i32).unsigned_abs();

    (dspin_min..=dspin_max)
        .step_by(2)
        .map(|s| {
            let s = HalfU32::from_doubled(s);
            get_spin_basis(s)
        })
        .flatten()
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinType {
    Fermionic,
    Bosonic,
}

pub struct SpinOperators;

impl SpinOperators {
    pub fn proj_z(spin: Braket<Spin>) -> f64 {
        if spin.bra == spin.ket {
            spin.bra.ms.value()
        } else {
            0.0
        }
    }

    pub fn ladder_plus(spin: Braket<Spin>) -> f64 {
        if spin.bra.s == spin.ket.s && spin.bra.ms.double_value() == spin.ket.ms.double_value() + 2
        {
            (spin.ket.s.value() * (spin.ket.s.value() + 1.) 
                - spin.bra.ms.value() * spin.ket.ms.value()).sqrt()
        } else {
            0.0
        }
    }

    pub fn ladder_minus(spin: Braket<Spin>) -> f64 {
        if spin.bra.s == spin.ket.s && spin.bra.ms.double_value() + 2 == spin.ket.ms.double_value()
        {
            (spin.ket.s.value() * (spin.ket.s.value() + 1.)
                - spin.bra.ms.value() * spin.ket.ms.value()).sqrt()
        } else {
            0.0
        }
    }

    pub fn dot(spin1: Braket<Spin>, spin2: Braket<Spin>) -> f64 {
        let val1 = Self::proj_z(spin1) * Self::proj_z(spin2);
        let val2 = 0.5 * Self::ladder_plus(spin1) * Self::ladder_minus(spin2);
        let val3 = 0.5 * Self::ladder_minus(spin1) * Self::ladder_plus(spin2);

        val1 + val2 + val3
    }

    /// Compute the Clebsch-Gordan coefficient <spin1; spin2 | spin3>.
    #[inline]
    pub fn clebsch_gordan(spin1: Spin, spin2: Spin, spin3: Spin) -> f64 {
        clebsch_gordan::clebsch_gordan(
            spin1.s, spin1.ms, 
            spin2.s, spin2.ms, 
            spin3.s, spin3.ms)
    }
}

#[cfg(test)]
#[cfg(feature = "faer")]
mod test {
    use clebsch_gordan::hu32;
    use faer::{assert_matrix_eq, mat, Mat};

    use crate::{
        cast_variant,
        states::{
            braket::Braket, operator::Operator, spins::{get_spin_basis, get_summed_spin_basis}, state::{into_variant, StateBasis}, States
        },
    };

    use super::{Spin, SpinOperators};

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum StateSep {
        Spin1(Spin),
        Spin2(Spin),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Combined {
        Spin(Spin),
    }

    #[test]
    fn test_spin_ops() {
        let mut state = States::default();

        let spin1 = into_variant(get_spin_basis(hu32!(1/2)), StateSep::Spin1);
        let spin2 = into_variant(get_spin_basis(hu32!(1/2)), StateSep::Spin2);

        state.push_state(StateBasis::new(spin1))
            .push_state(StateBasis::new(spin2));

        let basis = state.get_basis();

        let op = Operator::<Mat<f64>>::from_mel(
            &basis,
            [StateSep::Spin1(Default::default()), StateSep::Spin2(Default::default())],
            |[s1_braket, s2_braket]| {
                let s1 = Braket {
                    bra: cast_variant!(s1_braket.bra, StateSep::Spin1),
                    ket: cast_variant!(s1_braket.ket, StateSep::Spin1)
                };

                let s2 = Braket {
                    bra: cast_variant!(s2_braket.bra, StateSep::Spin2),
                    ket: cast_variant!(s2_braket.ket, StateSep::Spin2)
                };

                SpinOperators::dot(s1, s2)
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
        let spins = StateBasis::new(into_variant(get_summed_spin_basis(hu32!(1/2), hu32!(1/2)), Combined::Spin));
        combined.push_state(spins);
        let basis_comb = combined.get_basis();
        println!("{basis_comb}");

        let op = Operator::<Mat<f64>>::get_transformation(&basis, &basis_comb, |sep, combined| {
            let s1 = cast_variant!(sep[0], StateSep::Spin1);
            let s2 = cast_variant!(sep[1], StateSep::Spin2);

            let Combined::Spin(s) = combined[0];

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
