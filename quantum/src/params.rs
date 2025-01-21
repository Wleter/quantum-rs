pub mod particle;
pub mod particle_factory;
pub mod particles;

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

/// Struct to hold internal parameters.
/// Used to store information about a particle and composition of particles.
#[derive(Default)]
pub struct Params {
    params: HashMap<TypeId, Box<dyn Any>>,
}

impl Params {
    /// Insert or replace unique parameter of type `T`.
    pub fn insert<T: Any + 'static>(&mut self, value: T) -> &mut Self {
        self.params.insert(TypeId::of::<T>(), Box::new(value));

        self
    }

    /// Removes parameter of type `T`.
    pub fn remove<T: 'static>(&mut self) {
        self.params.remove(&TypeId::of::<T>());
    }

    /// Returns the reference of parameter of type `T` with given name if it exists.
    pub fn get<T: Any + 'static>(&self) -> Option<&T> {
        self.params
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// Returns the mutable reference of parameter of type `T` with given name if it exists.
    pub fn get_mut<T: Any + 'static>(&mut self) -> Option<&mut T> {
        self.params
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| value.downcast_mut::<T>())
    }
}
