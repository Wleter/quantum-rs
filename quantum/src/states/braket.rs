#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Braket<T> {
    pub bra: T,
    pub ket: T,
}

impl<T: PartialEq> Braket<T> {
    pub fn is_diagonal(&self) -> bool {
        self.ket == self.bra
    }
}

pub fn kron_delta<T: PartialEq, const N: usize>(
    brakets: [Braket<T>; N],
) -> f64 {
    if brakets.iter().all(|x| x.is_diagonal()) {
        1.0
    } else {
        0.0
    }
}
