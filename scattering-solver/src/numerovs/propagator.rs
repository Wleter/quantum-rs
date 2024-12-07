use std::marker::PhantomData;

use super::numerov_modifier::PropagatorModifier;

pub trait PropagatorData {
    fn crossed_distance(&self, r: f64) -> bool;

    fn step_size(&self) -> f64;

    fn current_g_func(&mut self);

    fn advance(&mut self);
}

pub trait MultiStep<D: PropagatorData> {
    /// Performs a step with the same step size
    fn step(&mut self, data: &mut D);

    /// Halves the step size without actually performing a step
    fn halve_step(&mut self, data: &mut D);

    /// Doubles the step size without actually performing a step
    fn double_step(&mut self, data: &mut D);
}

pub enum StepAction {
    Keep,
    Double,
    Halve,
}

pub trait StepRule<D: PropagatorData> {
    fn get_step(&self, data: &D) -> f64;

    fn assign(&mut self, data: &D) -> StepAction;
}

pub struct SingleStepRule {
    pub(crate) step: f64
}

impl<D: PropagatorData> StepRule<D> for SingleStepRule {
    fn get_step(&self, _data: &D) -> f64 {
        self.step
    }
    
    fn assign(&mut self, data: &D) -> StepAction {
        let prop_step = data.step_size();

        if prop_step > 1.2 * self.step {
            StepAction::Halve
        } else if prop_step < 2. * self.step {
            StepAction::Double
        } else {
            StepAction::Keep
        }
    }
}

pub struct MultiStepRule<D> {
    pub(super) wave_step_ratio: f64,

    pub(super) min_step: f64,
    pub(super) max_step: f64,

    pub(super) doubled_step: bool,
    phantom: PhantomData<D>,
}

impl<D> Default for MultiStepRule<D> {
    fn default() -> Self {
        Self { 
            doubled_step: false,
            min_step: 0., 
            max_step: f64::MAX,
            phantom: PhantomData,
            wave_step_ratio: 500.,
        }
    }
}

impl<D> MultiStepRule<D> {
    pub fn new(min_step: f64, max_step: f64, wave_step_ratio: f64) -> Self {
        Self {
            doubled_step: false,
            min_step, 
            max_step,
            phantom: PhantomData,
            wave_step_ratio,
        }
    }
}

/// Struct storing the result of a Numerov propagation
#[derive(Debug, Clone, Default)]
pub struct NumerovResult<T> {
    /// last position in the propagation
    pub r_last: f64,
    /// last step size
    pub dr: f64,
    /// Ratio of wave function on position r and r - dr
    pub wave_ratio: T,
}

pub struct Numerov<D, S, M> 
where 
    D: PropagatorData,
    S: StepRule<D>,
    M: MultiStep<D>
{
    pub data: D,
    pub(crate) step_rules: S,
    pub(crate) multi_step: M,
}

impl<D, S, M> Numerov<D, S, M> 
where 
    D: PropagatorData,
    S: StepRule<D>,
    M: MultiStep<D>
{
    pub fn propagate_to(&mut self, r: f64) {
        while !self.data.crossed_distance(r) {
            self.variable_step();
        }
    }

    pub fn propagate_to_with<P: PropagatorModifier<D>>(&mut self, r: f64, modifier: &mut P) {
        modifier.before(&mut self.data, r);
        
        while !self.data.crossed_distance(r) {
            self.variable_step();
            modifier.after_step(&mut self.data);
        }

        modifier.after_prop(&mut self.data);
    }

    pub fn variable_step(&mut self) {
        self.data.current_g_func();
        let mut action = self.step_rules.assign(&self.data);

        if let StepAction::Double = action {
            self.multi_step.double_step(&mut self.data);
            self.data.current_g_func();
        }

        let mut halved = false;
        while let StepAction::Halve = action {
            self.multi_step.halve_step(&mut self.data);
            action = self.step_rules.assign(&self.data);
            halved = true;
        };

        if halved {
            self.data.current_g_func();
        }

        self.multi_step.step(&mut self.data);
    }

    pub fn change_step_rule<S2: StepRule<D>>(self, step_rules: S2) -> Numerov<D, S2, M> {
        Numerov {
            data: self.data,
            step_rules: step_rules,
            multi_step: self.multi_step
        }
    }
}
