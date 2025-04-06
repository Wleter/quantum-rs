use faer::Mat;

use crate::{propagator::Equation, utility::AngMomentum};

#[derive(Clone, Debug)]
pub struct Boundary<T> {
    pub r_start: f64,
    pub direction: Direction,
    pub start_value: T,
    pub before_value: T,
}

// todo! change to R: Repr instead of T
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

    pub fn new_exponential_vanishing(r_start: f64, eq: &Equation<Mat<f64>>) -> Self {
        let size = eq.asymptotic.channel_energies.len();

        let mut start_mat = Mat::<f64>::identity(size, size);
        start_mat
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(eq.asymptotic(r_start))
            .for_each(|(s, e)| {
                let k = (2. * eq.mass * (eq.energy - e)).abs().sqrt();

                *s = -k
            });

        Self {
            r_start,
            direction: Direction::Inwards,
            start_value: start_mat.clone(),
            before_value: start_mat,
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
