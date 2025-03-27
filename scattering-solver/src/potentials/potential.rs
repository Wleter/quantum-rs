pub trait Dimension {
    fn size(&self) -> usize;
}

impl Dimension for f64 {
    fn size(&self) -> usize {
        1
    }
}

use faer::{Entity, Mat};
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

impl<P: Potential<Space = f64>> SimplePotential for P {}

pub trait MatPotential: Potential<Space = Mat<f64>> {}

impl<P: Potential<Space = Mat<f64>>> MatPotential for P {}
// impl SimplePotential for &dyn Potential<Space = f64> {}

#[derive(Clone)]
pub struct ScaledPotential<P: SimplePotential> {
    pub potential: P,
    pub scaling: f64,
}

impl<P: SimplePotential> Potential for ScaledPotential<P> {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        self.potential.value_inplace(r, value);
        *value *= self.scaling
    }

    fn size(&self) -> usize {
        1
    }
}

impl<P: SimplePotential + SubPotential> SubPotential for ScaledPotential<P> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        *value += self.scaling * self.potential.value(r)
    }
}
