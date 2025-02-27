use num::traits::Zero;
use std::{mem::discriminant, ops::Deref};

use super::{StatesBasis, StatesElement, braket::Braket};

#[derive(Debug, Clone)]
pub struct Operator<M> {
    backed: M,
}

impl<M> Operator<M> {
    pub fn new(mat: M) -> Self {
        Self { backed: mat }
    }

    pub fn into_backed(self) -> M {
        self.backed
    }
}

/// Cast the expression `value` to the variant `pat` or panic if it is mismatched.
/// # Syntax
/// - `cast_variant!($value, $pat)`
#[macro_export]
macro_rules! cast_variant {
    ($value:expr, $pat:path) => {{
        if let $pat(a) = $value {
            a
        } else {
            panic!("In correct variant cast")
        }
    }};
}

/// Cast the expression `value` to the variant `pat` or panic if it is mismatched.
/// # Syntax
/// - `cast_variant!($value, $pat)`
#[macro_export]
macro_rules! cast_variants {
    ($($args:ident: $states:path),* $(,)?) => {
        $(
            let $args = $crate::cast_variant!($args, $states);
        )*
    };
}

/// Cast the expression `value` braket to the variant `pat` or panic if it is mismatched.
/// # Syntax
/// - `cast_braket!($value, $pat)`
#[macro_export]
macro_rules! cast_braket {
    ($value:expr, $pat:path) => {{
        let bra = $crate::cast_variant!($value.bra, $pat);
        let ket = $crate::cast_variant!($value.ket, $pat);

        $crate::states::braket::Braket { bra, ket }
    }};
}

#[macro_export]
macro_rules! operator_mel {
    ($basis:expr, |[$($args:ident: $states:path),*]| $body:expr) => {
        Operator::<faer::Mat<f64>>::from_mel($basis, [$($states(Default::default())),*], |[$($args),*]| {
            $(
                let $args = $crate::cast_braket!($args, $states);
            )*

            $body
        })
    };
}

#[macro_export]
macro_rules! operator_diagonal_mel {
    ($basis:expr, |[$($args:ident: $states:path),*]| $body:expr) => {
        Operator::<faer::Mat<f64>>::from_diagonal_mel($basis, [$($states(Default::default())),*], |[$($args),*]| {
            $crate::cast_variants!($($args: $states),*);

            $body
        })
    };
}

fn get_mel<'a, const N: usize, T, F, E>(
    elements: &'a StatesBasis<T>,
    action_states: &[T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([Braket<T>; N]) -> E + 'a,
    T: Copy + PartialEq,
    E: Zero,
{
    let first = elements
        .first()
        .unwrap_or_else(|| panic!("0 size states basis")); // same variants for other elements

    let indices = action_states.map(|s| {
        first
            .iter()
            .enumerate()
            .find(|&(_, &x)| discriminant(&x) == discriminant(&s)) // variants are distinct by creation in States
            .map_or_else(|| panic!("action state not found in elements"), |x| x.0)
    });

    let diagonal_indices: Vec<usize> = (0..first.len()).filter(|x| !indices.contains(x)).collect();

    move |i, j| unsafe {
        let elements_i = elements.get_unchecked(i);
        let elements_j = elements.get_unchecked(j);

        for &index in &diagonal_indices {
            if elements_i.get_unchecked(index) != elements_j.get_unchecked(index) {
                return E::zero();
            }
        }

        let brakets = indices.map(|index| {
            let bra = *elements_i.get_unchecked(index);
            let ket = *elements_j.get_unchecked(index);

            Braket { bra, ket }
        });

        mat_element(brakets)
    }
}

fn get_diagonal_mel<'a, const N: usize, T, F, E>(
    elements: &'a StatesBasis<T>,
    action_states: &[T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([T; N]) -> E + 'a,
    T: Copy + PartialEq,
    E: Zero,
{
    let first = elements
        .first()
        .unwrap_or_else(|| panic!("0 size states basis")); // same variants for other elements

    let indices = action_states.map(|s| {
        first
            .iter()
            .enumerate()
            .find(|&(_, &x)| discriminant(&x) == discriminant(&s)) // variants are distinct by creation in States
            .map_or_else(|| panic!("action state not found in elements"), |x| x.0)
    });

    move |i, j| {
        if i != j {
            return E::zero();
        }

        unsafe {
            let elements_i = elements.get_unchecked(i);

            let ket = indices.map(|index| *elements_i.get_unchecked(index));

            mat_element(ket)
        }
    }
}

fn get_transformation<'a, T1, T2, F, E>(
    elements: &'a StatesBasis<T1>,
    elements_transformed: &'a StatesBasis<T2>,
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut(&StatesElement<T1>, &StatesElement<T2>) -> E + 'a,
    T1: Copy + PartialEq,
    T2: Copy + PartialEq,
    E: Zero,
{
    move |i, j| unsafe {
        let elements_i = elements_transformed.get_unchecked(i);
        let elements_j = elements.get_unchecked(j);

        mat_element(elements_j, elements_i)
    }
}

#[cfg(feature = "faer")]
use faer::{Entity, Mat};

#[cfg(feature = "faer")]
impl<E: Entity + Zero> Operator<Mat<E>> {
    pub fn from_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<T>; N]) -> E,
    {
        let mel = get_mel(elements, &action_states, mat_element);
        let mat = Mat::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([T; N]) -> E,
    {
        let mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = Mat::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, T2, F>(
        elements: &'a StatesBasis<T1>,
        elements_transformed: &'a StatesBasis<T2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1>, &StatesElement<T2>) -> E + 'a,
        T1: Copy + PartialEq,
        T2: Copy + PartialEq,
    {
        let mel = get_transformation(elements, elements_transformed, mat_element);
        let mat = Mat::from_fn(elements_transformed.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[cfg(feature = "faer")]
impl<E: Entity> Deref for Operator<Mat<E>> {
    type Target = Mat<E>;

    fn deref(&self) -> &Self::Target {
        &self.backed
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "nalgebra")]
impl<E: nalgebra::Scalar + Zero> Operator<DMatrix<E>> {
    pub fn from_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<T>; N]) -> E,
    {
        let mel = get_mel(elements, &action_states, mat_element);
        let mat = DMatrix::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([T; N]) -> E,
    {
        let mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = DMatrix::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, T2, F>(
        elements: &'a StatesBasis<T1>,
        elements_transformed: &'a StatesBasis<T2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1>, &StatesElement<T2>) -> E + 'a,
        T1: Copy + PartialEq,
        T2: Copy + PartialEq,
    {
        let mel = get_transformation(elements, elements_transformed, mat_element);
        let mat = DMatrix::from_fn(elements_transformed.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[cfg(feature = "nalgebra")]
impl<E> Deref for Operator<DMatrix<E>> {
    type Target = DMatrix<E>;

    fn deref(&self) -> &Self::Target {
        &self.backed
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, E: nalgebra::Scalar + Zero> Operator<SMatrix<E, N, N>> {
    pub fn from_mel<const M: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; M],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<T>; M]) -> E,
    {
        assert!(
            N < 10,
            "For larger matrices use DMatrix backed matrices instead"
        );
        assert!(
            N == elements.len(),
            "Elements does not have the same size as static matrix size"
        );

        let mel = get_mel(elements, &action_states, mat_element);
        let mat = SMatrix::from_fn(mel);

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const M: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; M],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([T; M]) -> E,
    {
        assert!(
            N < 10,
            "For larger matrices use DMatrix backed matrices instead"
        );
        assert!(
            N == elements.len(),
            "Elements does not have the same size as static matrix size"
        );

        let mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = SMatrix::from_fn(mel);

        Self { backed: mat }
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, E> Deref for Operator<SMatrix<E, N, N>> {
    type Target = SMatrix<E, N, N>;

    fn deref(&self) -> &Self::Target {
        &self.backed
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<E: Zero> Operator<Array2<E>> {
    pub fn from_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<T>; N]) -> E,
    {
        let mut mel = get_mel(elements, &action_states, mat_element);
        let mat = Array2::from_shape_fn((elements.len(), elements.len()), |(i, j)| mel(i, j));

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, F>(
        elements: &StatesBasis<T>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([T; N]) -> E,
    {
        let mut mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = Array2::from_shape_fn((elements.len(), elements.len()), |(i, j)| mel(i, j));

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, T2, F>(
        elements: &'a StatesBasis<T1>,
        elements_transformed: &'a StatesBasis<T2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1>, &StatesElement<T2>) -> E + 'a,
        T1: Copy + PartialEq,
        T2: Copy + PartialEq,
    {
        let mut mel = get_transformation(elements, elements_transformed, mat_element);
        let mat = Array2::from_shape_fn((elements_transformed.len(), elements.len()), |(i, j)| {
            mel(i, j)
        });

        Self { backed: mat }
    }
}

#[cfg(feature = "ndarray")]
impl<E> Deref for Operator<Array2<E>> {
    type Target = Array2<E>;

    fn deref(&self) -> &Self::Target {
        &self.backed
    }
}

#[cfg(test)]
mod test {
    use super::Operator;
    use crate::states::{States, state::StateBasis};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum StateIds {
        ElectronSpin((u32, i32)),
        Vibrational(i32),
    }

    fn prepare_states() -> States<StateIds> {
        let mut states = States::default();

        let e_state = StateBasis::new(vec![
            StateIds::ElectronSpin((2, -2)),
            StateIds::ElectronSpin((2, 0)),
            StateIds::ElectronSpin((2, 2)),
            StateIds::ElectronSpin((0, 0)),
        ]);
        states.push_state(e_state);

        let vib = StateBasis::new(vec![StateIds::Vibrational(-1), StateIds::Vibrational(-2)]);
        states.push_state(vib);

        states
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_faer_operator() {
        use faer::{Mat, mat};

        let elements = prepare_states().get_basis();

        let operator = Operator::<Mat<f64>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(Default::default())],
            |[el_state]| {
                let ket = el_state.ket;

                match ket {
                    StateIds::ElectronSpin(val) => val.1 as f64,
                    StateIds::Vibrational(val) => val as f64,
                }
            },
        );

        let expected = mat![
            [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
        ];
        assert_eq!(expected, operator.backed);

        let operator = Operator::<Mat<f64>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(Default::default())],
            |[el_state]| {
                let bra = el_state.bra;

                match bra {
                    StateIds::ElectronSpin(val) => val.1 as f64,
                    StateIds::Vibrational(val) => val as f64,
                }
            },
        );

        let expected = mat![
            [-2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(expected, operator.backed);

        let operator = Operator::<Mat<f64>>::from_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            |[el_state, vib]| {
                if vib.ket != vib.bra {
                    let ket_spin = cast_variant!(el_state.ket, StateIds::ElectronSpin);
                    let bra_spin = cast_variant!(el_state.bra, StateIds::ElectronSpin);

                    ((ket_spin.0 * 1000 + bra_spin.0 * 100) as i32 + ket_spin.1 * 10 + bra_spin.1)
                        as f64
                } else {
                    0.0
                }
            },
        );

        let expected = mat![
            [0.0, 0.0, 0.0, 0.0, 2178.0, 2198.0, 2218.0, 198.0],
            [0.0, 0.0, 0.0, 0.0, 2180.0, 2200.0, 2220.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, 2182.0, 2202.0, 2222.0, 202.0],
            [0.0, 0.0, 0.0, 0.0, 1980.0, 2000.0, 2020.0, 0.0],
            [2178.0, 2198.0, 2218.0, 198.0, 0.0, 0.0, 0.0, 0.0],
            [2180.0, 2200.0, 2220.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            [2182.0, 2202.0, 2222.0, 202.0, 0.0, 0.0, 0.0, 0.0],
            [1980.0, 2000.0, 2020.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(expected, operator.backed);

        let operator = Operator::<Mat<f64>>::from_diagonal_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            |[el_state, vib]| {
                let spin = cast_variant!(el_state, StateIds::ElectronSpin);
                let vib = cast_variant!(vib, StateIds::Vibrational);

                ((spin.0 as i32 * 100 + spin.1) as f64) * (-1.0f64).powi(vib)
            },
        );

        let expected = mat![
            [-198.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -202.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 198.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 202.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(expected, operator.backed);
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_faer_operators_cached() {
        use crate::{cached_mel, make_cache};
        use faer::{Mat, mat};

        let elements = prepare_states().get_basis();

        let operator = make_cache!(
            cache,
            Operator::<Mat<f64>>::from_mel(
                &elements,
                [
                    StateIds::ElectronSpin(Default::default()),
                    StateIds::Vibrational(Default::default())
                ],
                cached_mel!(cache, |[el_state, vib]| {
                    let ket_spin = cast_variant!(el_state.ket, StateIds::ElectronSpin);
                    let bra_spin = cast_variant!(el_state.bra, StateIds::ElectronSpin);

                    if vib.ket != vib.bra {
                        ((ket_spin.0 * 1000 + bra_spin.0 * 100) as i32
                            + ket_spin.1 * 10
                            + bra_spin.1) as f64
                    } else {
                        0.0
                    }
                })
            )
        );

        let expected = mat![
            [0.0, 0.0, 0.0, 0.0, 2178.0, 2198.0, 2218.0, 198.0],
            [0.0, 0.0, 0.0, 0.0, 2180.0, 2200.0, 2220.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, 2182.0, 2202.0, 2222.0, 202.0],
            [0.0, 0.0, 0.0, 0.0, 1980.0, 2000.0, 2020.0, 0.0],
            [2178.0, 2198.0, 2218.0, 198.0, 0.0, 0.0, 0.0, 0.0],
            [2180.0, 2200.0, 2220.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            [2182.0, 2202.0, 2222.0, 202.0, 0.0, 0.0, 0.0, 0.0],
            [1980.0, 2000.0, 2020.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(expected, operator.backed);
    }

    #[test]
    #[cfg(all(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
    fn test_operators() {
        use faer::Mat;
        use nalgebra::{DMatrix, SMatrix};
        use ndarray::Array2;

        use crate::states::braket::Braket;

        let elements = prepare_states().get_basis();

        let matrix_elements = |[el_state, vib]: [Braket<StateIds>; 2]| {
            if vib.ket != vib.bra {
                let ket_spin = cast_variant!(el_state.ket, StateIds::ElectronSpin);
                let bra_spin = cast_variant!(el_state.bra, StateIds::ElectronSpin);

                ((ket_spin.0 * 1000 + bra_spin.0 * 100) as i32 + ket_spin.1 * 10 + bra_spin.1)
                    as f64
            } else {
                0.0
            }
        };

        let operator_faer = Operator::<Mat<f64>>::from_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            matrix_elements,
        );
        let operator_d_matrix = Operator::<DMatrix<f64>>::from_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            matrix_elements,
        );
        let operator_s_matrix = Operator::<SMatrix<f64, 8, 8>>::from_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            matrix_elements,
        );
        let operator_ndarray = Operator::<Array2<f64>>::from_mel(
            &elements,
            [
                StateIds::ElectronSpin(Default::default()),
                StateIds::Vibrational(Default::default()),
            ],
            matrix_elements,
        );

        let faer_slice: Vec<f64> = operator_faer
            .col_iter()
            .map(|c| c.try_as_slice().unwrap().to_owned())
            .flatten()
            .collect();

        assert_eq!(&faer_slice, operator_d_matrix.backed.as_slice());
        assert_eq!(
            operator_d_matrix.backed.as_slice(),
            operator_s_matrix.backed.as_slice()
        );
        assert_eq!(
            operator_d_matrix.backed.transpose().as_slice(),
            operator_ndarray.backed.as_slice().unwrap()
        ); // transpose since the memory layout is different for ndarray
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_transformations() {
        use faer::{Mat, assert_matrix_eq, mat};

        use crate::states::StatesElement;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Combined {
            Spin((u32, i32)),
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Separated {
            Spin1((u32, i32)),
            Spin2((u32, i32)),
        }

        let mut states_combined = States::default();

        let el = StateBasis::new(vec![
            Combined::Spin((0, 0)),
            Combined::Spin((2, -2)),
            Combined::Spin((2, 0)),
            Combined::Spin((2, 2)),
        ]);
        states_combined.push_state(el);

        let elements_combined = states_combined
            .iter_elements()
            .filter(|x| matches!(x[0], Combined::Spin((2, _))))
            .collect();

        let mut states_sep = States::default();
        let s1 = StateBasis::new(vec![Separated::Spin1((1, -1)), Separated::Spin1((1, 1))]);
        let s2 = StateBasis::new(vec![Separated::Spin2((1, -1)), Separated::Spin2((1, 1))]);

        states_sep.push_state(s1).push_state(s2);
        let elements_sep = states_sep.get_basis();

        let transformation = |sep: &StatesElement<Separated>,
                              combined: &StatesElement<Combined>| {
            let (_, m1) = cast_variant!(sep[0], Separated::Spin1);
            let (_, m2) = cast_variant!(sep[1], Separated::Spin2);

            let Combined::Spin((s_comb, m_comb)) = combined[0];

            if m_comb == m1 + m2 {
                if m_comb == 0 {
                    let sign = if s_comb == 0 && m1 == -1 && m2 == 1 {
                        -1.
                    } else {
                        1.
                    };
                    sign * 0.5f64.sqrt()
                } else {
                    1.
                }
            } else {
                0.0
            }
        };

        let transformation_faer = Operator::<Mat<f64>>::get_transformation(
            &elements_sep,
            &elements_combined,
            transformation,
        );

        let expected = mat![
            [1.000, 0.000, 0.000, 0.000],
            [0.000, 0.707, 0.707, 0.000],
            [0.000, 0.000, 0.000, 1.000],
        ];

        assert_matrix_eq!(
            expected,
            transformation_faer.into_backed(),
            comp = abs,
            tol = 1e-3
        );
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_operator_shorthand() {
        use faer::{Mat, mat};

        let elements = prepare_states().get_basis();

        let operator: Operator<Mat<f64>> = operator_mel!(
            &elements,
            |[el: StateIds::ElectronSpin, vib: StateIds::Vibrational]| {
                if vib.ket != vib.bra {
                    ((el.ket.0 * 1000 + el.bra.0 * 100) as i32 + el.ket.1 * 10 + el.bra.1)
                        as f64
                } else {
                    0.0
                }
            }
        );

        let expected = mat![
            [0.0, 0.0, 0.0, 0.0, 2178.0, 2198.0, 2218.0, 198.0],
            [0.0, 0.0, 0.0, 0.0, 2180.0, 2200.0, 2220.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, 2182.0, 2202.0, 2222.0, 202.0],
            [0.0, 0.0, 0.0, 0.0, 1980.0, 2000.0, 2020.0, 0.0],
            [2178.0, 2198.0, 2218.0, 198.0, 0.0, 0.0, 0.0, 0.0],
            [2180.0, 2200.0, 2220.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            [2182.0, 2202.0, 2222.0, 202.0, 0.0, 0.0, 0.0, 0.0],
            [1980.0, 2000.0, 2020.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(expected, operator.backed);
    }
}
