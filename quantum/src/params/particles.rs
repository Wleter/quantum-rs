use std::ops::{Deref, DerefMut};

use crate::{
    params::Params,
    units::{
        Au,
        energy_units::{Energy, EnergyUnit},
        mass_units::Mass,
    },
};

use super::particle::Particle;

/// Struct to hold information about a particle composition.
#[derive(Clone)]
pub struct Particles {
    particles: Vec<Particle>,
    params: Params,
}

impl Particles {
    /// Creates two particle composition with given collision energy inserted inside `internals` as "energy".
    pub fn new_pair(
        first_particle: Particle,
        second_particle: Particle,
        energy: Energy<impl EnergyUnit>,
    ) -> Self {
        let mass1 = first_particle.get::<Mass<Au>>().unwrap().value();
        let mass2 = second_particle.get::<Mass<Au>>().unwrap().value();

        let inverse_reduced_mass: f64 = 1.0 / mass1 + 1.0 / mass2;

        let mut params = Params::default();
        params
            .insert(energy.to(Au))
            .insert(Mass(1. / inverse_reduced_mass, Au));

        Self {
            particles: vec![first_particle, second_particle],
            params,
        }
    }

    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    pub fn particles(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    /// Creates a particle composition given a vector of particles.
    pub fn new_custom(particles: Vec<Particle>) -> Self {
        let inverse_reduced_mass = particles.iter().fold(0.0, |acc, particle| {
            acc + 1.0 / particle.get::<Mass<Au>>().unwrap().value()
        });

        let mut params = Params::default();
        params.insert(Mass(1. / inverse_reduced_mass, Au));

        Self { particles, params }
    }

    /// Gets the reduced mass.
    pub fn red_mass(&self) -> f64 {
        self.params.get::<Mass<Au>>().unwrap().value()
    }
}

impl Deref for Particles {
    type Target = Params;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl DerefMut for Particles {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.params
    }
}
