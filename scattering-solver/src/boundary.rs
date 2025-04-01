use faer::Mat;

use crate::utility::AngMomentum;

#[derive(Clone, Debug)]
pub struct Boundary<T> {
    pub r_start: f64,
    pub direction: Direction,
    pub start_value: T,
    pub before_value: T,
}

impl<T> Boundary<T> {
    pub fn new(r_start: f64, direction: Direction, values: (T, T)) -> Self {
        Self {
            r_start,
            direction,
            start_value: values.0,
            before_value: values.1,
        }
    }
}

impl Boundary<f64> {
    pub fn new_vanishing(r_start: f64, direction: Direction) -> Self {
        Self {
            r_start,
            direction,
            start_value: 1e5,
            before_value: 1e10,
        }
    }
}

impl Boundary<Mat<f64>> {
    pub fn new_multi_vanishing(r_start: f64, direction: Direction, size: usize) -> Self {
        let id = Mat::<f64>::identity(size, size);

        Self {
            r_start,
            direction,
            start_value: 1e5 * &id,
            before_value: 1e10 * id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Inwards,
    Outwards,
    Step(f64),
}

#[derive(Clone, Debug)]
pub struct Asymptotic {
    pub centrifugal: Vec<AngMomentum>,
    pub entrance: usize,
    pub channel_energies: Vec<f64>,
    pub channel_states: Mat<f64>,
}

impl Asymptotic {
    pub fn single_default() -> Self {
        Self {
            centrifugal: vec![AngMomentum(0)],
            entrance: 0,
            channel_energies: vec![0.],
            channel_states: Mat::ones(1, 1),
        }
    }
}
