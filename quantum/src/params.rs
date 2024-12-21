pub mod particle;
pub mod particle_factory;
pub mod particles;

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use downcast_rs::{impl_downcast, DowncastSync};

pub trait CloneAny: DowncastSync {
    fn clone_any(&self) -> Box<dyn CloneAny>;
}
impl_downcast!(sync CloneAny);

impl<T: Any + Clone + Send + Sync> CloneAny for T {
    fn clone_any(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneAny> {
    fn clone(&self) -> Self {
        self.clone_any()
    }
}

/// Struct to hold internal parameters.
/// Used to store information about a particle and composition of particles.
#[derive(Default, Clone)]
pub struct Params {
    params: HashMap<TypeId, Box<dyn CloneAny>>,
}

impl Params {
    /// Insert or replace unique parameter of type `T`.
    pub fn insert<T: CloneAny + 'static>(&mut self, value: T) -> &mut Self {
        self.params.insert(TypeId::of::<T>(), Box::new(value));

        self
    }

    /// Removes parameter of type `T`.
    pub fn remove<T: 'static>(&mut self) {
        self.params.remove(&TypeId::of::<T>());
    }

    /// Returns the reference of parameter of type `T` with given name if it exists.
    pub fn get<T: CloneAny + 'static>(&self) -> Option<&T> {
        self.params
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// Returns the mutable reference of parameter of type `T` with given name if it exists.
    pub fn get_mut<T: CloneAny + 'static>(&mut self) -> Option<&mut T> {
        self.params
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| value.downcast_mut::<T>())
    }
}
