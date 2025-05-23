pub mod distance_units;
pub mod energy_units;
pub mod mass_units;

pub use distance_units::*;
pub use energy_units::*;
pub use mass_units::*;

/// Trait for units that can be converted to atomic units.
pub trait Unit: Copy {
    const TO_AU_MUL: f64;

    fn to_au(&self, value: f64) -> f64 {
        value * Self::TO_AU_MUL
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Au;

impl Unit for Au {
    const TO_AU_MUL: f64 = 1.0;
}
