use crate::units::mass_units::{Dalton, Mass};

use super::particle::Particle;

#[derive(Clone, Copy, Debug)]
pub struct RotConst(pub f64);

pub fn create_atom(name: &str) -> Option<Particle> {
    let mass = match name {
        "Ne" => Mass(20.1797, Dalton),
        "Li6" => Mass(6.015122, Dalton),
        "Li7" => Mass(7.016004, Dalton),
        "Na23" => Mass(22.989770, Dalton),
        "K40" => Mass(39.963707, Dalton),
        "Rb85" => Mass(84.911789, Dalton),
        "Rb87" => Mass(86.90918053, Dalton),
        "Cs133" => Mass(132.905447, Dalton),
        _ => return None,
    };

    Some(Particle::new(name, mass))
}

pub fn create_molecule(name: &str) -> Option<Particle> {
    let mass = match name {
        "OCS" => Mass(60.07, Dalton),
        _ => return None,
    };

    let rot_const = match name {
        "OCS" => 9.243165268327e-7,
        _ => return None,
    };

    let mut particle = Particle::new(name, mass);
    particle.params.insert(RotConst(rot_const));

    Some(particle)
}
