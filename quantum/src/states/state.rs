#[derive(Clone, Debug)]
pub struct State<T, V> {
    pub(crate) variant: T,
    pub(crate) basis: Vec<V>,
}

impl<T: Copy, V: Copy> State<T, V> {
    pub fn new(variant: T, basis: Vec<V>) -> Self {
        assert!(!basis.is_empty(), "0 size basis is not allowed");

        Self { variant, basis }
    }
}

impl<T, V> State<T, V> {
    pub fn size(&self) -> usize {
        self.basis.len()
    }
}
