use super::StatesElement;

pub struct Braket<T, V> {
    pub ket: StatesElement<T, V>,
    pub bra: StatesElement<T, V>,
}

pub struct StateBraket<T, V> {
    pub ket: (T, V),
    pub bra: (T, V),
}

impl<T: PartialEq, V: PartialEq> StateBraket<T, V> {
    pub fn is_diagonal(&self) -> bool {
        self.ket == self.bra
    }
}

pub fn dirac_delta<T: PartialEq, V: PartialEq, const N: usize>(brakets: [StateBraket<T, V>; N]) -> f64 {
    if brakets.iter().all(|x| x.is_diagonal()) {
        1.0
    } else {
        0.0
    }
}