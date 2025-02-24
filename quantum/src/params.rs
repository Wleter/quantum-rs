pub mod particle;
pub mod particle_factory;
pub mod particles;

use downcast_rs::{DowncastSync, impl_downcast};
use dyn_clone::DynClone;
use std::{any::TypeId, collections::HashMap};

pub trait CloneAny: DynClone + DowncastSync {}
impl_downcast!(sync CloneAny);
dyn_clone::clone_trait_object!(CloneAny);

impl<T: Clone + Send + Sync + 'static> CloneAny for T {}

/// Struct to hold internal parameters.
/// Used to store information about a particle and composition of particles.
#[derive(Default, Clone)]
pub struct Params {
    params: HashMap<TypeId, Box<dyn CloneAny>>,
}

impl Params {
    /// Insert or replace unique parameter of type `T`.
    pub fn insert<T: CloneAny>(&mut self, value: T) -> &mut Self {
        self.params.insert(TypeId::of::<T>(), Box::new(value));

        self
    }

    /// Removes parameter of type `T`.
    pub fn remove<T: 'static>(&mut self) {
        self.params.remove(&TypeId::of::<T>());
    }

    /// Returns the reference of parameter of type `T` with given name if it exists.
    pub fn get<T: CloneAny>(&self) -> Option<&T> {
        self.params
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// Returns the mutable reference of parameter of type `T` with given name if it exists.
    pub fn get_mut<T: CloneAny>(&mut self) -> Option<&mut T> {
        self.params
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| value.downcast_mut::<T>())
    }
}

#[cfg(test)]
mod test {
    use super::Params;

    #[derive(Clone)]
    struct FloatValue(f64);

    #[derive(Clone)]
    struct IntValue(i32);

    #[test]
    pub fn test_params() {
        let mut params = Params::default();

        params.insert(FloatValue(1.0));

        let mut params_cloned = params.clone();
        params_cloned.insert(IntValue(2));

        assert!(params.get::<FloatValue>().is_some_and(|x| x.0 == 1.0));
        assert!(
            params_cloned
                .get::<FloatValue>()
                .is_some_and(|x| x.0 == 1.0)
        );

        assert!(params.get::<IntValue>().is_none());
        assert!(params_cloned.get::<IntValue>().is_some_and(|x| x.0 == 2));
    }
}
