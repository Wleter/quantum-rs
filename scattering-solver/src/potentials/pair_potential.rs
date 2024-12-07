use super::potential::{Potential, SubPotential};

/// Composite potential that has value equal to the sum of its components
#[derive(Debug, Clone)]
pub struct PairPotential<P: SubPotential, V: SubPotential> {
    first: P,
    second: V
}

impl<P: SubPotential, V: SubPotential> PairPotential<P, V> {
    pub fn new(first: P, second: V) -> Self {
        assert!(first.size() == second.size());

        Self {
            first,
            second,
        }
    }
}

impl<P, V, T> Potential for PairPotential<P, V> 
where
    P: SubPotential<Space = T>, 
    V: SubPotential<Space = T>,
{
    type Space = T;
    
    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        self.first.value_inplace(r, value);

        self.second.value_add(r, value);
    }
    
    fn size(&self) -> usize {
        self.first.size()
    }
}

impl<P, V, T> SubPotential for PairPotential<P, V> 
where
    P: SubPotential<Space = T>, 
    V: SubPotential<Space = T>,
{
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        self.first.value_add(r, value);
        self.second.value_add(r, value);
    }
}
