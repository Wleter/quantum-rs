use super::potential::{Potential, SubPotential};

/// Potential of the form d0 * r^n
#[derive(Debug, Clone)]
pub struct Dispersion {
    pub d0: f64,
    pub n: i32,
}

impl Dispersion {
    pub fn new(d0: f64, n: i32) -> Self {
        Self { d0: d0, n, }
    }
}

impl Potential for Dispersion {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        *value = self.d0 * r.powi(self.n);
    }
    
    fn size(&self) -> usize {
        1
    }
}

impl SubPotential for Dispersion {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        *value += self.d0 * r.powi(self.n);
    }
}
