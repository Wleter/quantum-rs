use quantum::units::{distance_units::Distance, energy_units::Energy, Au};

use crate::utility::AngularSpin;

use super::{composite_potential::Composite, dispersion_potential::Dispersion};

/// Creates a Lennard-Jones potential with given parameters
pub fn create_lj(d6: Energy<Au>, r6: Distance<Au>) -> Composite<Dispersion> {
    let d6 = d6.to_au();
    let r6 = r6.to_au();
    let c12 = d6 * r6.powi(12);
    let c6 = -2.0 * d6 * r6.powi(6);

    let mut potential = Composite::new(Dispersion::new(c12, -12));
    potential.add_potential(Dispersion::new(c6, -6));

    potential
}

/// Creates a Lennard-Jones potential with given parameters
pub fn create_centrifugal(red_mass: f64, l: AngularSpin) -> Option<Dispersion> {
    if l.0 == 0 {
        return None
    }

    Some(Dispersion::new((l.0 * (l.0 + 1)) as f64 / (2. * red_mass), -2))
}