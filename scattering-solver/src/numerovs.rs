use std::{f64::consts::PI, marker::PhantomData};

use faer::Mat;

use crate::propagator::{Equation, MultiStep, Repr, Solution};

pub mod multi_numerov;
pub mod propagator_watcher;
pub mod single_numerov;
// pub mod bound_numerov;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ratio<T>(pub T);
impl<T> Repr<T> for Ratio<T> {}

#[derive(Clone)]
pub struct Numerov<'a, T, R, M, S>
where
    M: MultiStep<T, R>,
    S: StepRule<T>,
    R: Repr<T>,
{
    pub(super) equation: Equation<'a, T>,
    pub(super) solution: Solution<R>,

    pub(super) multi_step: M,
    pub(super) step_rule: S,
    pub(super) phantom: PhantomData<T>,
}

impl<'a, T, M, S, R> Numerov<'a, T, R, M, S>
where
    M: MultiStep<T, R>,
    S: StepRule<T>,
    R: Repr<T>,
{
    pub fn new_general(eq: Equation<'a, T>, sol: Solution<R>, multi_step: M, step_rule: S) -> Self {
        Self {
            equation: eq,
            solution: sol,
            multi_step,
            step_rule,
            phantom: PhantomData,
        }
    }

    pub fn equation(&self) -> &Equation<'a, T> {
        &self.equation
    }

    pub fn solution(&self) -> &Solution<R> {
        &self.solution
    }
}

impl<'a, T, M, S, R> Numerov<'a, T, R, M, S>
where
    M: MultiStep<T, R>,
    S: StepRule<T>,
    R: Repr<T>,
{
    pub fn change_step_rule<S2: StepRule<T>>(self, step_rule: S2) -> Numerov<'a, T, R, M, S2> {
        Numerov {
            equation: self.equation,
            solution: self.solution,
            multi_step: self.multi_step,
            step_rule,
            phantom: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum StepAction {
    Keep,
    Double,
    Halve,
}

pub trait StepRule<T> {
    fn get_step(&self, w_matrix: &T) -> f64;

    fn step_action(&mut self, current_dr: f64, w_matrix: &T) -> StepAction;
}

#[derive(Clone, Debug)]
pub struct SingleStepRule {
    pub step: f64,
}

impl<T> StepRule<T> for SingleStepRule {
    fn get_step(&self, _w_matrix: &T) -> f64 {
        self.step
    }

    fn step_action(&mut self, current_dr: f64, _w_matrix: &T) -> StepAction {
        let current_dr = current_dr.abs();

        if current_dr > 1.2 * self.step {
            StepAction::Halve
        } else if current_dr < 0.5 * self.step {
            StepAction::Double
        } else {
            StepAction::Keep
        }
    }
}

#[derive(Clone)]
pub struct LocalWavelengthStepRule {
    pub(super) wave_step_ratio: f64,

    pub(super) min_step: f64,
    pub(super) max_step: f64,

    pub(super) doubled_step: bool,
}

impl Default for LocalWavelengthStepRule {
    fn default() -> Self {
        Self {
            doubled_step: false,
            min_step: 0.,
            max_step: f64::MAX,
            wave_step_ratio: 500.,
        }
    }
}

impl LocalWavelengthStepRule {
    pub fn new(min_step: f64, max_step: f64, wave_step_ratio: f64) -> Self {
        Self {
            doubled_step: false,
            min_step,
            max_step,
            wave_step_ratio,
        }
    }
}

impl StepRule<f64> for LocalWavelengthStepRule {
    fn get_step(&self, w_matrix: &f64) -> f64 {
        let lambda = 2. * PI / w_matrix.abs().sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }

    fn step_action(&mut self, current_dr: f64, w_matrix: &f64) -> StepAction {
        let dr = current_dr.abs();
        let step = self.get_step(w_matrix);

        if dr > 1.2 * step {
            self.doubled_step = false;
            StepAction::Halve
        } else if dr < 0.5 * step && !self.doubled_step {
            self.doubled_step = true;
            StepAction::Double
        } else {
            self.doubled_step = false;
            StepAction::Keep
        }
    }
}

impl StepRule<Mat<f64>> for LocalWavelengthStepRule {
    fn get_step(&self, w_matrix: &Mat<f64>) -> f64 {
        let max_g_val = w_matrix
            .diagonal()
            .column_vector()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let lambda = 2. * PI / max_g_val.abs().sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }

    fn step_action(&mut self, current_dr: f64, w_matrix: &Mat<f64>) -> StepAction {
        let dr = current_dr.abs();
        let step = self.get_step(w_matrix);

        if dr > 1.2 * step {
            self.doubled_step = false;
            StepAction::Halve
        } else if dr < 0.5 * step && !self.doubled_step {
            self.doubled_step = true;
            StepAction::Double
        } else {
            self.doubled_step = false;
            StepAction::Keep
        }
    }
}
