use std::marker::PhantomData;

use super::potential::{Potential, SubPotential};
#[derive(Debug, Clone)]
pub struct Diagonal<A, P: Potential> {
    potentials: Vec<P>,
    phantom: PhantomData<A>
}

impl<A, P: Potential> Diagonal<A, P> {
    pub fn from_vec(potentials: Vec<P>) -> Self {
        Self { 
            potentials,
            phantom: PhantomData
        }
    }
}

use faer::Mat;

impl<P: Potential<Space = f64>> Potential for Diagonal<Mat<f64>, P> {
    type Space = Mat<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Mat<f64>) {
        value.fill_zero();

        value.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.potentials.iter())
            .for_each(|(val, p)| p.value_inplace(r, val))
    }
    
    fn size(&self) -> usize {
        self.potentials.len()
    }
}

impl<P: SubPotential<Space = f64>> SubPotential for Diagonal<Mat<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Mat<f64>) {
        value.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.potentials.iter())
            .for_each(|(val, p)| p.value_add(r, val))
    }
}
