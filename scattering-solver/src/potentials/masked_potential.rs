use super::potential::{Dimension, Potential, SimplePotential, SubPotential};
use faer::{Mat, zip, unzip};

#[derive(Debug, Clone)]
pub struct MaskedPotential<M, P: Potential> {
    potential: P,
    masking: M,
}

impl<M, P: Potential> MaskedPotential<M, P> {
    pub fn new(potential: P, masking: M) -> Self {
        Self { potential, masking }
    }

    pub fn masking(&self) -> &M {
        &self.masking
    }
}

impl<P: Potential<Space = f64>> Potential for MaskedPotential<Mat<f64>, P> {
    type Space = Mat<f64>;

    fn value_inplace(&self, r: f64, value: &mut Mat<f64>) {
        let potential_value = self.potential.value(r);

        zip!(value.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v = potential_value * m;
        });
    }

    fn size(&self) -> usize {
        self.masking.size()
    }
}

impl<P: Potential<Space = f64>> SubPotential for MaskedPotential<Mat<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Mat<f64>) {
        let potential_value = self.potential.value(r);

        zip!(value.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v += potential_value * m;
        });
    }
}
