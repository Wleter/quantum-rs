use std::{marker::PhantomData, ops::AddAssign};

use num::Zero;

use super::potential::{Dimension, Potential, SubPotential};

/// Potential that gives `value` according to provided function.
#[derive(Clone)]
pub struct FunctionPotential<T, F: Fn(f64, &mut T)> {
    function: F,
    phantom: PhantomData<T>
}

impl<T: Clone, F: Fn(f64, &mut T)> FunctionPotential<T, F> {
    /// Creates new function potential with given function.
    pub fn new(function: F) -> Self {
        Self { 
            function,
            phantom: PhantomData
        }
    }
}

impl<T: Dimension + Zero, F> Potential for FunctionPotential<T, F>
where
    F: Fn(f64, &mut T)
{
    type Space = T;
    
    fn value_inplace(&self, r: f64, value: &mut T) {
        (self.function)(r, value)
    }

    fn size(&self) -> usize {
        panic!("cannot determine size of the FunctionPotential")
    }
}

impl<T: Zero + AddAssign + Dimension, F> SubPotential for FunctionPotential<T, F>
where
    F: Fn(f64, &mut T)
{
    fn value_add(&self, r: f64, value: &mut T) {
        let mut holder = T::zero();

        (self.function)(r, &mut holder);

        *value += holder;
    }
}