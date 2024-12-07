use std::{
    iter::Peekable,
    mem::{discriminant, Discriminant},
    slice::Iter,
};

use super::state::State;

#[derive(Clone, Debug)]
pub enum StateType<T, V> {
    Irreducible(State<T, V>),
    Sum(Vec<State<T, V>>),
}

impl<T, V> StateType<T, V> {
    pub fn size(&self) -> usize {
        match self {
            StateType::Irreducible(irreducible_states) => irreducible_states.size(),
            StateType::Sum(sum_states) => sum_states.iter().fold(0, |acc, s| acc + s.size()),
        }
    }

    pub fn iter(&self) -> StateTypeIter<'_, T, V> {
        match self {
            StateType::Irreducible(s) => StateTypeIter {
                state_type: self,
                sum_iter: Iter::default().peekable(),
                irreducible_iter: s.basis.iter(),
            },
            StateType::Sum(vec_s) => StateTypeIter {
                state_type: self,
                sum_iter: vec_s.iter().peekable(),
                irreducible_iter: vec_s
                    .first()
                    .unwrap_or_else(|| panic!("no states to iter"))
                    .basis
                    .iter(),
            },
        }
    }

    pub fn discriminant(&self) -> Option<Discriminant<T>> {
        match self {
            StateType::Irreducible(irreducible_states) => {
                Some(discriminant(&irreducible_states.variant))
            }
            StateType::Sum(vec) => {
                let mut iterator = vec.iter().map(|x| discriminant(&x.variant));

                let first = iterator
                    .next()
                    .unwrap_or_else(|| panic!("0 sized state is not allowed"));
                if iterator.all(|x| x == first) {
                    Some(first)
                } else {
                    None
                }
            }
        }
    }
}

pub struct StateTypeIter<'a, T, V> {
    state_type: &'a StateType<T, V>,
    sum_iter: Peekable<Iter<'a, State<T, V>>>,
    irreducible_iter: Iter<'a, V>,
}

impl<'a, T, K> Iterator for StateTypeIter<'a, T, K> {
    type Item = (&'a T, &'a K);

    fn next(&mut self) -> Option<Self::Item> {
        match self.state_type {
            StateType::Irreducible(s) => self.irreducible_iter.next().map(|v| (&s.variant, v)),
            StateType::Sum(_) => match self.irreducible_iter.next() {
                Some(val) => Some((&self.sum_iter.peek().unwrap().variant, val)),
                None => {
                    self.sum_iter.next().unwrap();
                    self.sum_iter.peek().and_then(|s| {
                        self.irreducible_iter = s.basis.iter();

                        self.irreducible_iter.next().map(|v| (&s.variant, v))
                    })
                }
            },
        }
    }
}
