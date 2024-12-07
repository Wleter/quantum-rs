use super::Unit;

/// Struct for representing distance unit values
/// # Examples
/// ```
/// use quantum::units::{Au, distance_units::{Distance, Angstrom}};
/// let distance_ang = Distance(1.0, Angstrom);
/// let distance_au = distance_ang.to(Au);
/// let distance = distance_ang.to_au();
/// assert!(distance == distance_au.value());
#[derive(Debug, Copy, Clone)]
pub struct Distance<U: Unit>(pub f64, pub U);

impl<U: Unit> Distance<U> {
    pub fn to_au(&self) -> f64 {
        self.1.to_au(self.0)
    }

    pub fn to<V: Unit>(&self, unit: V) -> Distance<V> {
        Distance(self.1.to_au(self.0) / unit.to_au(1.0), unit)
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }
}

#[derive(Copy, Clone)]
pub struct Angstrom;

impl Unit for Angstrom {
    const TO_AU_MUL: f64 = 1.88973;
}
