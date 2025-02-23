use super::{Au, Unit};

pub trait EnergyUnit: Unit {}

/// Struct for representing energy unit values
/// # Examples
/// ```
/// use quantum::units::energy_units::{Energy, Kelvin, CmInv};
/// let energy_kelvin = Energy(1.0, Kelvin);
/// let energy_cm_inv = energy_kelvin.to(CmInv);
/// let energy = energy_kelvin.to_au();
#[derive(Debug, Copy, Clone)]
pub struct Energy<U: EnergyUnit>(pub f64, pub U);

impl<U: EnergyUnit> Energy<U> {
    pub fn to_au(&self) -> f64 {
        self.1.to_au(self.0)
    }

    pub fn to<V: EnergyUnit>(&self, unit: V) -> Energy<V> {
        Energy(self.1.to_au(self.0) / unit.to_au(1.0), unit)
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }
}

impl EnergyUnit for Au {}

#[derive(Copy, Clone)]
pub struct Kelvin;

impl Unit for Kelvin {
    const TO_AU_MUL: f64 = 3.1668105e-6;
}
impl EnergyUnit for Kelvin {}

#[derive(Copy, Clone)]
pub struct CmInv;

impl Unit for CmInv {
    const TO_AU_MUL: f64 = 4.5563352812e-6;
}
impl EnergyUnit for CmInv {}

#[derive(Copy, Clone)]
pub struct MHz;

impl Unit for MHz {
    const TO_AU_MUL: f64 = 1.51982850071586e-10;
}
impl EnergyUnit for MHz {}

#[derive(Copy, Clone)]
pub struct GHz;

impl Unit for GHz {
    const TO_AU_MUL: f64 = 1.51982850071586e-07;
}
impl EnergyUnit for GHz {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn energy_units() {
        let energy_kelvin = Energy(1.0, Kelvin);
        let energy_cm_inv = energy_kelvin.to(CmInv);
        let energy_from_kelvin = energy_kelvin.to_au();
        let energy_from_cm_inv = energy_cm_inv.to_au();
        assert_eq!(energy_from_kelvin, energy_from_cm_inv);
        assert!(energy_cm_inv.value() > 0.6950);
        assert!(energy_cm_inv.value() < 0.6951);
    }
}
