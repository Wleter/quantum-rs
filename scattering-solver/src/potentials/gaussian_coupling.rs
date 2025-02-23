use quantum::units::energy_units::{Energy, EnergyUnit};

use super::potential::{Potential, SubPotential};

/// Gaussian coupling potential
#[derive(Clone)]
pub struct GaussianCoupling {
    width: f64,
    center: f64,
    strength: f64,
}

impl GaussianCoupling {
    /// Creates new Gaussian coupling potential with given strength, center position and width
    pub fn new(strength: Energy<impl EnergyUnit>, center: f64, width: f64) -> Self {
        Self {
            width,
            center,
            strength: strength.to_au(),
        }
    }
}

impl Potential for GaussianCoupling {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        *value = self.strength * (-((r - self.center) / self.width).powi(2) / 2.0).exp()
    }
    
    fn size(&self) -> usize {
        1
    }
}

impl SubPotential for GaussianCoupling {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        *value += self.strength * (-((r - self.center) / self.width).powi(2) / 2.0).exp()
    }
}