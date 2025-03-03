use std::{marker::PhantomData, time::{Duration, Instant}};

use crate::{potentials::potential::{MatPotential, SimplePotential}, utility::inverse_inplace_det};
use super::{multi_numerov::MultiNumerovData, numerov_modifier::NumerovLogging, propagator::{MultiStep, Numerov, PropagatorData, StepRule}, single_numerov::SingleNumerovData};

pub struct DummyMultiStep<D: PropagatorData> {
    _phantom: PhantomData<D>
}

impl<D: PropagatorData> Default for DummyMultiStep<D> {
    fn default() -> Self {
        Self { _phantom: Default::default() }
    }
}

impl<'a, P, S> Numerov<MultiNumerovData<'a, P>, S, DummyMultiStep<MultiNumerovData<'a, P>>>
where
    P: MatPotential,
    S: StepRule<MultiNumerovData<'a, P>>,
{
    pub fn estimate_propagation_duration(&mut self, r_end: f64) -> (Duration, u64) {
        let mut logging = NumerovLogging::default();
        self.propagate_to_with(r_end, &mut logging);
        let steps_no = logging.steps_no();

        let start = Instant::now();
        for _ in 0..10 { 
            self.data.psi2_det = inverse_inplace_det(
                self.data.psi1.as_ref(),
                self.data.psi2.as_mut(),
                &mut self.data.perm_buffer,
                &mut self.data.perm_inv_buffer,
            );
        }
        let end = start.elapsed();

        let full_duration = 2 * (steps_no as f32 / 10.) as u32 * end;

        (full_duration, steps_no)
    }
}

impl<P: SimplePotential> MultiStep<SingleNumerovData<'_, P>> for DummyMultiStep<SingleNumerovData<'_, P>> {
    fn step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        data.r += data.dr
    }

    fn halve_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        data.dr /= 2.
    }

    fn double_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        data.dr *= 2.
    }
}

impl<P: MatPotential> MultiStep<MultiNumerovData<'_, P>> for DummyMultiStep<MultiNumerovData<'_, P>> {
    fn step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        data.r += data.dr
    }

    fn halve_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        data.dr /= 2.
    }

    fn double_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        data.dr *= 2.
    }
}