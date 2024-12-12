use std::marker::PhantomData;

use super::potential::{Potential, SubPotential};

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Debug, Clone)]
pub struct MultiCoupling<A, P: Potential> {
    potentials: Vec<(P, usize, usize)>,
    size: usize,
    phantom: PhantomData<A>,
}

impl<A, P: Potential> MultiCoupling<A, P>
{
    /// Creates new multi coupling potential with given vector of potentials with their coupling indices in potential matrix.
    /// If `symmetric` is true, the coupling matrix will be symmetric.
    pub fn new(size: usize, potentials: Vec<(P, usize, usize)>) -> Self {
        for p in &potentials {
            assert!(p.1 < size);
            assert!(p.2 < size);
        }
        
        Self {
            potentials,
            size,
            phantom: PhantomData
        }
    }

    pub fn new_neighboring(couplings: Vec<P>) -> Self {
        let size = couplings.len() + 1;

        let numbered_potentials = couplings
            .into_iter()
            .enumerate()
            .map(|(i, potential)| (potential, i, i + 1))
            .collect();

        Self::new(size, numbered_potentials)
    }
}

use faer::Mat;

impl<P: Potential<Space = f64>> Potential for MultiCoupling<Mat<f64>, P> {
    type Space = Mat<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Mat<f64>) {
        value.fill_zero();
        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut(*i, *j));
        }

        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut(*j, *i));
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<P: SubPotential<Space = f64>> SubPotential for MultiCoupling<Mat<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Mat<f64>) {
        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut(*i, *j));
        }

        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut(*j, *i));
        }
    }
}
