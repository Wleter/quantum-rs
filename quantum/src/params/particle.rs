use std::ops::{Deref, DerefMut};

use crate::{
    params::Params,
    units::{mass_units::Mass, mass_units::MassUnit, Au},
};

/// Struct to hold information about a particle.
/// To create a predefined particle use [`crate::particle_factory`].
#[derive(Default, Clone)]
pub struct Particle {
    name: String,
    params: Params,
}

impl Particle {
    /// Creates new particle with given name and mass
    pub fn new(name: &str, mass: Mass<impl MassUnit>) -> Self {
        let mut params = Params::default();
        params.insert(mass.to(Au));

        Particle {
            name: name.to_string(),
            params,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Deref for Particle {
    type Target = Params;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl DerefMut for Particle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.params
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        params::particle_factory,
        units::{
            mass_units::{Dalton, Mass},
            Au,
        },
    };

    #[derive(Clone, Copy)]
    struct Parameter(u32);

    #[test]
    fn particle() {
        let particle = particle_factory::create_atom("Ne");
        assert!(particle.is_some());
        let mut particle = particle.unwrap();

        let mass = particle.get::<Mass<Au>>().copied();
        assert!(mass.is_some());
        assert_eq!(mass.unwrap().to_au(), Mass(20.1797, Dalton).to_au());

        particle.insert(Parameter(32));
        let parameter = particle.get::<Parameter>();
        assert!(parameter.is_some());
        assert_eq!(parameter.unwrap().0, 32);

        let particle = particle_factory::create_atom("Non existing atom");
        assert!(particle.is_none());
    }
}
