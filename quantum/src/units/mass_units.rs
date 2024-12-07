use super::Unit;

/// Struct for representing mass unit values
/// # Examples
/// ```
/// use quantum::units::{Au, mass_units::{Mass, Dalton}};
/// let mass_dalton = Mass(1.0, Dalton);
/// let mass_au = mass_dalton.to(Au);
/// let mass = mass_dalton.to_au();
/// assert_eq!(mass, mass_au.value())
#[derive(Debug, Copy, Clone)]
pub struct Mass<U: Unit>(pub f64, pub U);

impl<U: Unit> Mass<U> {
    pub fn to_au(&self) -> f64 {
        self.1.to_au(self.0)
    }

    pub fn to<V: Unit>(&self, unit: V) -> Mass<V> {
        Mass(self.1.to_au(self.0) / unit.to_au(1.0), unit)
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }
}

#[derive(Copy, Clone)]
pub struct Dalton;

impl Unit for Dalton {
    const TO_AU_MUL: f64 = 1822.88839;
}
