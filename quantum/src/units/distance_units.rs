use super::{Au, Unit};

pub trait DistanceUnit: Unit {}

/// Struct for representing distance unit values
/// # Examples
/// ```
/// use quantum::units::*;
/// let distance_ang = Distance(1.0, Angstrom);
/// let distance_au = distance_ang.to(Au);
/// let distance = distance_ang.to_au();
/// assert!(distance == distance_au.value());
#[derive(Debug, Copy, Clone)]
pub struct Distance<U: DistanceUnit>(pub f64, pub U);

impl<U: DistanceUnit> Distance<U> {
    pub fn to_au(&self) -> f64 {
        self.1.to_au(self.0)
    }

    pub fn to<V: DistanceUnit>(&self, unit: V) -> Distance<V> {
        Distance(self.1.to_au(self.0) / unit.to_au(1.0), unit)
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }
}

impl DistanceUnit for Au {}

#[derive(Copy, Clone)]
pub struct Angstrom;

impl Unit for Angstrom {
    const TO_AU_MUL: f64 = 1. / 0.529177210544;
}
impl DistanceUnit for Angstrom {}
