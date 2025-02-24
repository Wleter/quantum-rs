pub mod braket;
pub mod state;

use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    slice::Iter,
};

use state::StateBasis;

#[cfg(any(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
pub mod operator;

#[cfg(feature = "spins")]
pub mod spins;

#[derive(Clone, Debug)]
pub struct States<T>(Vec<StateBasis<T>>);

impl<T> Default for States<T> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<T> States<T> {
    pub fn size(&self) -> usize {
        self.0.iter().fold(1, |acc, s| acc * s.size())
    }

    pub fn push_state(&mut self, state: StateBasis<T>) -> &mut Self {
        let variant = state.variant();

        if self.0.iter().any(|x| x.variant() == variant) {
            panic!("Each state has to have unique variant type");
        }

        self.0.push(state);

        self
    }
}

impl<T: Copy> States<T> {
    pub fn iter_elements(&self) -> StatesIter<'_, T> {
        StatesIter {
            states: &self.0,
            states_iter: self.0.iter().map(|s| s.elements().iter()).collect(),
            current: StatesElement(Vec::with_capacity(self.0.len())),
            current_index: 0,
            size: self.size(),
        }
    }

    pub fn get_basis(&self) -> StatesBasis<T> {
        self.iter_elements().collect()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StatesElement<T>(pub Vec<T>);

impl<T> Deref for StatesElement<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for StatesElement<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Debug> Display for StatesElement<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in self.iter() {
            write!(f, "|{s:?} ⟩ ")?
        }

        Ok(())
    }
}

pub struct StatesIter<'a, T> {
    states: &'a [StateBasis<T>],
    states_iter: Vec<Iter<'a, T>>,
    current: StatesElement<T>,
    current_index: usize,
    size: usize,
}

impl<T: Copy> Iterator for StatesIter<'_, T> {
    type Item = StatesElement<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            return None;
        }
        if self.current_index == 0 {
            for s in self.states_iter.iter_mut() {
                let s_curr = s.next().unwrap(); // at least 1 element exists

                self.current.0.push(*s_curr);
            }
            self.current_index += 1;

            return Some(self.current.clone());
        }

        for ((s_spec, s), s_type) in self
            .current
            .iter_mut()
            .zip(self.states_iter.iter_mut())
            .zip(self.states.iter())
        {
            match s.next() {
                Some(s_spec_new) => {
                    *s_spec = *s_spec_new;
                    break;
                }
                None => {
                    *s = s_type.elements().iter();
                    let s_curr = s.next().unwrap(); // at least 1 element exists
                    *s_spec = *s_curr;
                }
            }
        }
        self.current_index += 1;

        Some(self.current.clone())
    }
}

#[derive(Debug, Clone)]
pub struct StatesBasis<T>(Vec<StatesElement<T>>);

impl<T> FromIterator<StatesElement<T>> for StatesBasis<T> {
    fn from_iter<I: IntoIterator<Item = StatesElement<T>>>(iter: I) -> Self {
        let mut elements = StatesBasis(vec![]);

        for val in iter {
            elements.0.push(val);
        }

        elements
    }
}

impl<T> IntoIterator for StatesBasis<T> {
    type Item = StatesElement<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> Deref for StatesBasis<T> {
    type Target = Vec<StatesElement<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for StatesBasis<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Debug> Display for StatesBasis<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for state in &self.0 {
            writeln!(f, "{state}")?
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use state::StateBasis;

    use super::*;

    #[allow(unused)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum StateIds {
        ElectronSpin((u32, i32)),
        NuclearSpin((u32, i32)),
        Vibrational(i32),
    }

    #[test]
    fn states_creation() {
        let mut states = States::default();

        let e_state = StateBasis::new(vec![
            StateIds::ElectronSpin((2, -2)),
            StateIds::ElectronSpin((2, 0)),
            StateIds::ElectronSpin((2, 2)),
            StateIds::ElectronSpin((0, 0)),
        ]);
        states.push_state(e_state);

        let nuclear = StateBasis::new(vec![
            StateIds::NuclearSpin((1, -1)),
            StateIds::NuclearSpin((1, 1)),
        ]);
        states.push_state(nuclear);

        let vib = StateBasis::new(vec![StateIds::Vibrational(-1), StateIds::Vibrational(-2)]);
        states.push_state(vib);

        let expected = "States([StateBasis { elements: [ElectronSpin((2, -2)), ElectronSpin((2, 0)), ElectronSpin((2, 2)), ElectronSpin((0, 0))], variant: Discriminant(0) }, \
                        StateBasis { elements: [NuclearSpin((1, -1)), NuclearSpin((1, 1))], variant: Discriminant(1) }, \
                        StateBasis { elements: [Vibrational(-1), Vibrational(-2)], variant: Discriminant(2) }])";

        assert_eq!(expected, format!("{:?}", states))
    }

    #[test]
    fn state_iteration() {
        let mut states = States::default();

        let e_state = StateBasis::new(vec![
            StateIds::ElectronSpin((2, -2)),
            StateIds::ElectronSpin((2, 0)),
            StateIds::ElectronSpin((2, 2)),
            StateIds::ElectronSpin((0, 0)),
        ]);
        states.push_state(e_state);

        let nuclear = StateBasis::new(vec![
            StateIds::NuclearSpin((1, -1)),
            StateIds::NuclearSpin((1, 1)),
        ]);
        states.push_state(nuclear);

        let vib = StateBasis::new(vec![StateIds::Vibrational(-1), StateIds::Vibrational(-2)]);
        states.push_state(vib);

        let expected: Vec<&str> = "\
StatesElement([ElectronSpin((2, -2)), NuclearSpin((1, -1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, 0)), NuclearSpin((1, -1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, 2)), NuclearSpin((1, -1)), Vibrational(-1)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, -1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, -2)), NuclearSpin((1, 1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, 0)), NuclearSpin((1, 1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, 2)), NuclearSpin((1, 1)), Vibrational(-1)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, 1)), Vibrational(-1)])
StatesElement([ElectronSpin((2, -2)), NuclearSpin((1, -1)), Vibrational(-2)])
StatesElement([ElectronSpin((2, 0)), NuclearSpin((1, -1)), Vibrational(-2)])
StatesElement([ElectronSpin((2, 2)), NuclearSpin((1, -1)), Vibrational(-2)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, -1)), Vibrational(-2)])
StatesElement([ElectronSpin((2, -2)), NuclearSpin((1, 1)), Vibrational(-2)])
StatesElement([ElectronSpin((2, 0)), NuclearSpin((1, 1)), Vibrational(-2)])
StatesElement([ElectronSpin((2, 2)), NuclearSpin((1, 1)), Vibrational(-2)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, 1)), Vibrational(-2)])"
            .split("\n")
            .collect();

        for (state, exp) in states.iter_elements().zip(expected.into_iter()) {
            assert_eq!(exp, format!("{:?}", state));
        }
    }

    #[test]
    fn state_filtering() {
        let mut states = States::default();

        let e_state = StateBasis::new(vec![
            StateIds::ElectronSpin((2, -2)),
            StateIds::ElectronSpin((2, 0)),
            StateIds::ElectronSpin((2, 2)),
            StateIds::ElectronSpin((0, 0)),
        ]);
        states.push_state(e_state);

        let nuclear = StateBasis::new(vec![
            StateIds::NuclearSpin((1, -1)),
            StateIds::NuclearSpin((1, 1)),
        ]);
        states.push_state(nuclear);

        let vib = StateBasis::new(vec![StateIds::Vibrational(-1), StateIds::Vibrational(-2)]);
        states.push_state(vib);

        let filtered: StatesBasis<StateIds> = states
            .iter_elements()
            .filter(|s| matches!(s[0], StateIds::ElectronSpin((0, _))))
            .collect();

        let expected: Vec<&str> = "\
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, -1)), Vibrational(-1)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, 1)), Vibrational(-1)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, -1)), Vibrational(-2)])
StatesElement([ElectronSpin((0, 0)), NuclearSpin((1, 1)), Vibrational(-2)])"
            .split("\n")
            .collect();

        for (state, &exp) in filtered.iter().zip(expected.iter()) {
            assert_eq!(exp, format!("{:?}", state));
        }
    }

    #[test]
    #[should_panic]
    fn test_wrong_state_initialization() {
        let mut states = States::default();

        let e_state = StateBasis::new(vec![
            StateIds::ElectronSpin((2, -2)),
            StateIds::ElectronSpin((2, 0)),
            StateIds::ElectronSpin((2, 2)),
            StateIds::ElectronSpin((0, 0)),
        ]);
        states.push_state(e_state);

        let nuclear = StateBasis::new(vec![
            StateIds::NuclearSpin((1, -1)),
            StateIds::NuclearSpin((1, 1)),
        ]);
        states.push_state(nuclear);

        let second_electron_spin = StateBasis::new(vec![
            StateIds::ElectronSpin((1, -1)),
            StateIds::ElectronSpin((1, 1)),
        ]);
        states.push_state(second_electron_spin);
    }
}
