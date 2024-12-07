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

#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Inwards,
    Outwards,
    Step(f64),
}