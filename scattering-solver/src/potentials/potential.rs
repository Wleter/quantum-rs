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
