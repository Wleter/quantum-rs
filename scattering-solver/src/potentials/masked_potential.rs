use super::potential::{Dimension, Potential, SimplePotential, SubPotential};

#[derive(Debug, Clone)]
pub struct MaskedPotential<M, P: Potential> {
    potential: P,
    masking: M
}

impl<M, P: Potential> MaskedPotential<M, P> {
    pub fn new(potential: P, masking: M) -> Self {
        Self { 
            potential,
            masking
        }
    }
}

use faer::unzipped;
use faer::{Mat, zipped};

impl<P: Potential<Space = f64>> Potential for MaskedPotential<Mat<f64>, P> {
    type Space = Mat<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Mat<f64>) {
        let potential_value = self.potential.value(r);

        zipped!(value.as_mut(), self.masking.as_ref())
            .for_each(|unzipped!(mut v, m)| {
                v.write(potential_value * m.read());
            });
    }
    
    fn size(&self) -> usize {
        self.masking.size()
    }
}

impl<P: Potential<Space = f64>> SubPotential for MaskedPotential<Mat<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Mat<f64>) {
        let potential_value = self.potential.value(r);

        zipped!(value.as_mut(), self.masking.as_ref())
            .for_each(|unzipped!(mut v, m)| {
                v.write(v.read() + potential_value * m.read());
            });
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<P: Potential<Space = f64>> Potential for MaskedPotential<Array2<f64>, P> {
    type Space = Array2<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Array2<f64>) {
        let potential_value = self.potential.value(r);

        value.zip_mut_with(&self.masking, |v, m| *v = potential_value * m);
    }
    
    fn size(&self) -> usize {
        self.masking.size()
    }
}

#[cfg(feature = "ndarray")]
impl<P: Potential<Space = f64>> SubPotential for MaskedPotential<Array2<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Array2<f64>) {
        let potential_value = self.potential.value(r);

        value.zip_mut_with(&self.masking, |v, m| *v += potential_value * m);
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "nalgebra")]
impl<P: Potential<Space = f64>> Potential for MaskedPotential<DMatrix<f64>, P> {
    type Space = DMatrix<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut DMatrix<f64>) {
        let potential_value = self.potential.value(r);

        value.zip_apply(&self.masking, |v, m| *v = potential_value * m);
    }

    fn size(&self) -> usize {
        self.masking.size()
    }
}

#[cfg(feature = "nalgebra")]
impl<P: Potential<Space = f64>> SubPotential for MaskedPotential<DMatrix<f64>, P> {
    fn value_add(&self, r: f64, value: &mut DMatrix<f64>) {
        let potential_value = self.potential.value(r);

        value.zip_apply(&self.masking, |v, m| *v += potential_value * m);
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: Potential<Space = f64>> Potential for MaskedPotential<SMatrix<f64, N, N>, P> {
    type Space = SMatrix<f64, N, N>;
    
    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        let potential_value = self.potential.value(r);

        value.zip_apply(&self.masking, |v, m| *v = potential_value * m);
    }
    
    fn size(&self) -> usize {
        self.masking.size()
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: Potential<Space = f64>> SubPotential for MaskedPotential<SMatrix<f64, N, N>, P> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        let potential_value = self.potential.value(r);

        value.zip_apply(&self.masking, |v, m| *v += potential_value * m);
    }
}