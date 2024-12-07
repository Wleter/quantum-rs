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

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<P: Potential<Space = f64>> Potential for Diagonal<Array2<f64>, P> {
    type Space = Array2<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Array2<f64>) {
        value.fill(0.);

        value.diag_mut()
            .iter_mut()
            .zip(self.potentials.iter())
            .for_each(|(val, p)| p.value_inplace(r, val))
    }
    
    fn size(&self) -> usize {
        self.potentials.len()
    }
}

#[cfg(feature = "ndarray")]
impl<P: SubPotential<Space = f64>> SubPotential for Diagonal<Array2<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Array2<f64>) {
        value.diag_mut()
            .iter_mut()
            .zip(self.potentials.iter())
            .for_each(|(val, p)| p.value_add(r, val))
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "nalgebra")]
impl<P: Potential<Space = f64>> Potential for Diagonal<DMatrix<f64>, P> {
    type Space = DMatrix<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut DMatrix<f64>) {
        value.fill(0.);

        for (i, p) in self.potentials.iter().enumerate() {
            p.value_inplace(r, value.get_mut((i, i)).unwrap());
        }
    }

    fn size(&self) -> usize {
        self.potentials.len()
    }
}

#[cfg(feature = "nalgebra")]
impl<P: SubPotential<Space = f64>> SubPotential for Diagonal<DMatrix<f64>, P> {
    fn value_add(&self, r: f64, value: &mut DMatrix<f64>) {
        for (i, p) in self.potentials.iter().enumerate() {
            p.value_add(r, value.get_mut((i, i)).unwrap());
        }
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: Potential<Space = f64>> Potential for Diagonal<SMatrix<f64, N, N>, P> {
    type Space = SMatrix<f64, N, N>;
    
    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        value.fill(0.);
        
        for (i, p) in self.potentials.iter().enumerate() {
            p.value_inplace(r, value.get_mut((i, i)).unwrap());
        }
    }
    
    fn size(&self) -> usize {
        self.potentials.len()
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: SubPotential<Space = f64>> SubPotential for Diagonal<SMatrix<f64, N, N>, P> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        for (i, p) in self.potentials.iter().enumerate() {
            p.value_add(r, value.get_mut((i, i)).unwrap());
        }
    }
}