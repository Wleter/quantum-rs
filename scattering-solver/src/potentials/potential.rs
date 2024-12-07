pub trait Dimension {
    fn size(&self) -> usize;
}

impl Dimension for f64 {
    fn size(&self) -> usize {
        1
    }
}

use faer::{Mat, Entity};
impl<T: Entity> Dimension for Mat<T> {
    fn size(&self) -> usize {
        assert!(self.nrows() == self.ncols());

        self.nrows()
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{SMatrix, DMatrix};

#[cfg(feature = "nalgebra")]
impl<T, const N: usize> Dimension for SMatrix<T, N, N> {
    fn size(&self) -> usize {
        self.nrows()
    }
}

#[cfg(feature = "nalgebra")]
impl<T> Dimension for DMatrix<T> {
    fn size(&self) -> usize {
        assert!(self.nrows() == self.ncols());

        self.nrows()
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<T> Dimension for Array2<T> {
    fn size(&self) -> usize {
        assert!(self.nrows() == self.ncols());

        self.nrows()
    }
}

/// Trait defining potential functionality
pub trait Potential {
    type Space;

    fn value_inplace(&self, r: f64, value: &mut Self::Space);

    fn size(&self) -> usize;
}

/// Trait defining potentials that can be part of the larger potential
pub trait SubPotential: Potential {
    fn value_add(&self, r: f64, value: &mut Self::Space);
}

pub trait SimplePotential: Potential<Space = f64> {
    fn value(&self, r: f64) -> f64 {
        let mut val = 0.;
        self.value_inplace(r, &mut val);

        val
    }
}

impl<P: Potential<Space = f64>> SimplePotential for P { }
