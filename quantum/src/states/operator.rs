use num::traits::Zero;
use std::{mem::discriminant, ops::Deref};

use super::{braket::StateBraket, StatesBasis, StatesElement};

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
            unreachable!()
        }
    }};
}

fn get_mel<'a, const N: usize, T, V, F, E>(
    elements: &'a StatesBasis<T, V>,
    action_states: &[T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([StateBraket<T, V>; N]) -> E + 'a,
    T: Copy + PartialEq,
    V: Copy + PartialEq,
    E: Zero,
{
    let first = elements
        .first()
        .unwrap_or_else(|| panic!("0 size states basis")); // same variants for other elements

    let indices = action_states.map(|s| {
        first
            .variants
            .iter()
            .enumerate()
            .find(|&(_, &x)| discriminant(&x) == discriminant(&s)) // variants are distinct by creation in States
            .map_or_else(|| panic!("action state not found in elements"), |x| x.0)
    });

    let diagonal_indices: Vec<usize> = (0..first.variants.len())
        .filter(|x| !indices.contains(x))
        .collect();

    move |i, j| unsafe {
        let elements_i = elements.get_unchecked(i);
        let elements_j = elements.get_unchecked(j);

        for &index in &diagonal_indices {
            if elements_i.variants.get_unchecked(index) != elements_j.variants.get_unchecked(index)
                || elements_i.values.get_unchecked(index) != elements_j.values.get_unchecked(index)
            {
                return E::zero();
            }
        }

        let brakets = indices.map(|index| {
            let bra = (
                *elements_i.variants.get_unchecked(index),
                *elements_i.values.get_unchecked(index),
            );

            let ket = (
                *elements_j.variants.get_unchecked(index),
                *elements_j.values.get_unchecked(index),
            );

            StateBraket { ket, bra }
        });

        mat_element(brakets)
    }
}

fn get_diagonal_mel<'a, const N: usize, T, V, F, E>(
    elements: &'a StatesBasis<T, V>,
    action_states: &[T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([(T, V); N]) -> E + 'a,
    T: Copy + PartialEq,
    V: Copy + PartialEq,
    E: Zero,
{
    let first = elements
        .first()
        .unwrap_or_else(|| panic!("0 size states basis")); // same variants for other elements

    let indices = action_states.map(|s| {
        first
            .variants
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

            let ket = indices.map(|index| {
                let ket = (
                    *elements_i.variants.get_unchecked(index),
                    *elements_i.values.get_unchecked(index),
                );

                ket
            });

            mat_element(ket)
        }
    }
}

fn get_transformation<'a, T1, V1, T2, V2, F, E>(
    elements: &'a StatesBasis<T1, V1>,
    elements_transformed: &'a StatesBasis<T2, V2>,
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut(&StatesElement<T1, V1>, &StatesElement<T2, V2>) -> E + 'a,
    T1: Copy + PartialEq,
    V1: Copy + PartialEq,
    T2: Copy + PartialEq,
    V2: Copy + PartialEq,
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
    pub fn from_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([StateBraket<T, V>; N]) -> E,
    {
        let mel = get_mel(elements, &action_states, mat_element);
        let mat = Mat::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([(T, V); N]) -> E,
    {
        let mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = Mat::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, V1, T2, V2, F>(
        elements: &'a StatesBasis<T1, V1>,
        elements_transformed: &'a StatesBasis<T2, V2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1, V1>, &StatesElement<T2, V2>) -> E + 'a,
        T1: Copy + PartialEq,
        V1: Copy + PartialEq,
        T2: Copy + PartialEq,
        V2: Copy + PartialEq,
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
    pub fn from_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([StateBraket<T, V>; N]) -> E,
    {
        let mel = get_mel(elements, &action_states, mat_element);
        let mat = DMatrix::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([(T, V); N]) -> E,
    {
        let mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = DMatrix::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, V1, T2, V2, F>(
        elements: &'a StatesBasis<T1, V1>,
        elements_transformed: &'a StatesBasis<T2, V2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1, V1>, &StatesElement<T2, V2>) -> E + 'a,
        T1: Copy + PartialEq,
        V1: Copy + PartialEq,
        T2: Copy + PartialEq,
        V2: Copy + PartialEq,
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
    pub fn from_mel<const M: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; M],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([StateBraket<T, V>; M]) -> E,
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

    pub fn from_diagonal_mel<const M: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; M],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([(T, V); M]) -> E,
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
    pub fn from_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([StateBraket<T, V>; N]) -> E,
    {
        let mut mel = get_mel(elements, &action_states, mat_element);
        let mat = Array2::from_shape_fn((elements.len(), elements.len()), |(i, j)| mel(i, j));

        Self { backed: mat }
    }

    pub fn from_diagonal_mel<const N: usize, T: Copy + PartialEq, V: Copy + PartialEq, F>(
        elements: &StatesBasis<T, V>,
        action_states: [T; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([(T, V); N]) -> E,
    {
        let mut mel = get_diagonal_mel(elements, &action_states, mat_element);
        let mat = Array2::from_shape_fn((elements.len(), elements.len()), |(i, j)| mel(i, j));

        Self { backed: mat }
    }

    pub fn get_transformation<'a, T1, V1, T2, V2, F>(
        elements: &'a StatesBasis<T1, V1>,
        elements_transformed: &'a StatesBasis<T2, V2>,
        mat_element: F,
    ) -> Self
    where
        F: FnMut(&StatesElement<T1, V1>, &StatesElement<T2, V2>) -> E + 'a,
        T1: Copy + PartialEq,
        V1: Copy + PartialEq,
        T2: Copy + PartialEq,
        V2: Copy + PartialEq,
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
    use crate::states::{state::State, state_type::StateType, States};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum StateIds {
        ElectronSpin(u32),
        Vibrational,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum ElementValues {
        Spin(i32),
        Vibrational(i32),
    }

    fn prepare_states() -> States<StateIds, ElementValues> {
        let mut states = States::default();

        let triplet_elements = vec![
            ElementValues::Spin(-2),
            ElementValues::Spin(0),
            ElementValues::Spin(2),
        ];
        let triplet = State::new(StateIds::ElectronSpin(2), triplet_elements);
        let singlet = State::new(StateIds::ElectronSpin(0), vec![ElementValues::Spin(0)]);

        let e_state = StateType::Sum(vec![triplet, singlet]);
        states.push_state(e_state);

        let vib_elements = vec![
            ElementValues::Vibrational(-1),
            ElementValues::Vibrational(-2),
        ];
        let vib = StateType::Irreducible(State::new(StateIds::Vibrational, vib_elements));
        states.push_state(vib);

        states
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_faer_operator() {
        use faer::{mat, Mat};

        let elements = prepare_states().get_basis();

        let operator =
            Operator::<Mat<f64>>::from_mel(&elements, [StateIds::ElectronSpin(0)], |[el_state]| {
                let ket = el_state.ket;

                match ket.1 {
                    ElementValues::Spin(val) => val as f64,
                    ElementValues::Vibrational(val) => val as f64,
                }
            });

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

        let operator =
            Operator::<Mat<f64>>::from_mel(&elements, [StateIds::ElectronSpin(0)], |[el_state]| {
                let bra = el_state.bra;

                match bra.1 {
                    ElementValues::Spin(val) => val as f64,
                    ElementValues::Vibrational(val) => val as f64,
                }
            });

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
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
            |[el_state, vib]| {
                if vib.ket != vib.bra {
                    let ket_spin = cast_variant!(el_state.ket.0, StateIds::ElectronSpin);
                    let bra_spin = cast_variant!(el_state.bra.0, StateIds::ElectronSpin);

                    let ket_spin_z = cast_variant!(el_state.ket.1, ElementValues::Spin);
                    let bra_spin_z = cast_variant!(el_state.bra.1, ElementValues::Spin);

                    ((ket_spin * 1000 + bra_spin * 100) as i32 + ket_spin_z * 10 + bra_spin_z)
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
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
            |[el_state, vib]| {
                let spin = cast_variant!(el_state.0, StateIds::ElectronSpin);
                let spin_z = cast_variant!(el_state.1, ElementValues::Spin);
                let vib = cast_variant!(vib.1, ElementValues::Vibrational);

                ((spin as i32 * 100 + spin_z) as f64) * (-1.0f64).powi(vib)
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
        use faer::{mat, Mat};

        let elements = prepare_states().get_basis();

        let operator = make_cache!(
            cache,
            Operator::<Mat<f64>>::from_mel(
                &elements,
                [StateIds::ElectronSpin(0), StateIds::Vibrational],
                cached_mel!(cache, |[el_state, vib]| {
                    let ket_spin = cast_variant!(el_state.ket.0, StateIds::ElectronSpin);
                    let bra_spin = cast_variant!(el_state.bra.0, StateIds::ElectronSpin);

                    let ket_spin_z = cast_variant!(el_state.ket.1, ElementValues::Spin);
                    let bra_spin_z = cast_variant!(el_state.bra.1, ElementValues::Spin);
                    if vib.ket != vib.bra {
                        ((ket_spin * 1000 + bra_spin * 100) as i32 + ket_spin_z * 10 + bra_spin_z)
                            as f64
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

        use crate::states::braket::StateBraket;

        let elements = prepare_states().get_basis();

        let matrix_elements = |[el_state, vib]: [StateBraket<StateIds, ElementValues>; 2]| {
            if vib.ket != vib.bra {
                let ket_spin = cast_variant!(el_state.ket.0, StateIds::ElectronSpin);
                let bra_spin = cast_variant!(el_state.bra.0, StateIds::ElectronSpin);

                let ket_spin_z = cast_variant!(el_state.ket.1, ElementValues::Spin);
                let bra_spin_z = cast_variant!(el_state.bra.1, ElementValues::Spin);

                ((ket_spin * 1000 + bra_spin * 100) as i32 + ket_spin_z * 10 + bra_spin_z) as f64
            } else {
                0.0
            }
        };

        let operator_faer = Operator::<Mat<f64>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
            matrix_elements,
        );
        let operator_d_matrix = Operator::<DMatrix<f64>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
            matrix_elements,
        );
        let operator_s_matrix = Operator::<SMatrix<f64, 8, 8>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
            matrix_elements,
        );
        let operator_ndarray = Operator::<Array2<f64>>::from_mel(
            &elements,
            [StateIds::ElectronSpin(0), StateIds::Vibrational],
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
        use faer::{assert_matrix_eq, mat, Mat};

        use crate::states::StatesElement;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Combined {
            Spin(u32),
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Separated {
            Spin1(u32),
            Spin2(u32),
        }

        let mut states_combined = States::default();
        let singlet = State::new(Combined::Spin(0), vec![0]);
        let triplet = State::new(Combined::Spin(2), vec![-2, 0, 2]);
        states_combined.push_state(StateType::Sum(vec![singlet, triplet]));

        let elements_combined = states_combined
            .iter_elements()
            .filter(|x| x.variants[0] == Combined::Spin(2))
            .collect();

        let mut states_sep = States::default();
        let s1 = State::new(Separated::Spin1(1), vec![-1, 1]);
        let s2 = State::new(Separated::Spin2(1), vec![-1, 1]);

        states_sep
            .push_state(StateType::Irreducible(s1))
            .push_state(StateType::Irreducible(s2));
        let elements_sep = states_sep.get_basis();

        let transformation =
            |sep: &StatesElement<Separated, i32>, combined: &StatesElement<Combined, i32>| {
                let m1 = sep.values[0];
                let m2 = sep.values[1];

                let Combined::Spin(s_comb) = combined.variants[0];
                let m_comb = combined.values[0];

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
}
