use std::mem::{discriminant, Discriminant};

#[derive(Clone, Debug)]
pub struct StateBasis<T> {
    elements: Vec<T>,
    variant: Discriminant<T>
}

impl<T> StateBasis<T> {
    pub fn new(elements: Vec<T>) -> Self {
        assert!(!elements.is_empty(), "0 size basis is not allowed");

        let variant = discriminant(elements.first().unwrap());
        assert!(
            elements.iter().all(|x| discriminant(x) == variant), 
            "only same variant types in basis is permitted"
        );

        Self { 
            elements,
            variant 
        }
    }

    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    pub fn variant(&self) -> &Discriminant<T> {
        &self.variant
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

pub fn into_variant<V, T>(elements: Vec<V>, variant: fn(V) -> T) -> Vec<T> {
    elements.into_iter().map(variant).collect()
}