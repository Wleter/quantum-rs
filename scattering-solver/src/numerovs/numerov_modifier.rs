use faer::Mat;

use crate::{
    observables::s_matrix::SingleSMatrix,
    potentials::potential::{MatPotential, SimplePotential},
};

use super::{
    multi_numerov::MultiNumerovData, propagator::PropagatorData, single_numerov::SingleNumerovData,
};

pub trait PropagatorModifier<D: PropagatorData> {
    fn before(&mut self, _data: &mut D, _r_stop: f64) {}

    fn after_step(&mut self, data: &mut D);

    fn after_prop(&mut self, _data: &mut D) {}
}

pub struct MultiPropagatorModifier<'a, D: PropagatorData> {
    modifiers: Vec<&'a mut dyn PropagatorModifier<D>>,
}

impl<'a, D: PropagatorData> MultiPropagatorModifier<'a, D> {
    pub fn new(modifiers: Vec<&'a mut dyn PropagatorModifier<D>>) -> Self {
        Self { modifiers }
    }
}

impl<D: PropagatorData> PropagatorModifier<D> for MultiPropagatorModifier<'_, D> {
    fn before(&mut self, data: &mut D, r_stop: f64) {
        for modifier in self.modifiers.iter_mut() {
            modifier.before(data, r_stop);
        }
    }

    fn after_step(&mut self, data: &mut D) {
        for modifier in self.modifiers.iter_mut() {
            modifier.after_step(data);
        }
    }

    fn after_prop(&mut self, data: &mut D) {
        for modifier in self.modifiers.iter_mut() {
            modifier.after_prop(data);
        }
    }
}

pub(super) enum SampleConfig {
    Each(usize),
    Step(f64),
}

pub struct WaveStorage<T> {
    pub rs: Vec<f64>,
    pub waves: Vec<T>,

    pub(super) last_value: T,
    pub(super) counter: usize,
    pub(super) capacity: usize,
    pub(super) sampling: SampleConfig,
}

impl<T: Clone> WaveStorage<T> {
    pub fn new(sampling: Sampling, wave_init: T, capacity: usize) -> Self {
        let rs = Vec::with_capacity(capacity);
        let mut waves = Vec::with_capacity(capacity);
        waves.push(wave_init.clone());

        let sampling = match sampling {
            Sampling::Uniform => SampleConfig::Step(0.),
            Sampling::Variable => SampleConfig::Each(1),
        };

        Self {
            rs,
            waves,
            last_value: wave_init,
            capacity,
            counter: 0,
            sampling,
        }
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for WaveStorage<f64>
where
    P: SimplePotential,
{
    fn before(&mut self, data: &mut SingleNumerovData<'_, P>, r_stop: f64) {
        if let SampleConfig::Step(value) = &mut self.sampling {
            *value = (data.r - r_stop).abs() / self.capacity as f64
        }

        self.rs.push(data.r);
    }

    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        self.last_value *= data.psi1;

        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value);
                }

                if self.rs.len() == self.capacity {
                    *sample_each *= 2;

                    self.rs = self
                        .rs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, r)| *r)
                        .collect();

                    self.waves = self
                        .waves
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, w)| *w)
                        .collect();
                }
            }
            SampleConfig::Step(sample_step) => {
                if (data.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value);
                }
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Sampling {
    Uniform,

    #[default]
    Variable,
}

pub struct ScatteringVsDistance<S> {
    pub(super) r_min: f64,
    pub(super) capacity: usize,
    pub(super) take_per: f64,

    pub distances: Vec<f64>,
    pub s_matrices: Vec<S>,
}

impl<S> ScatteringVsDistance<S> {
    pub fn new(r_min: f64, capacity: usize) -> Self {
        Self {
            r_min,
            capacity,
            take_per: 0.,
            distances: Vec::with_capacity(capacity),
            s_matrices: Vec::with_capacity(capacity),
        }
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for ScatteringVsDistance<SingleSMatrix>
where
    P: SimplePotential,
{
    fn before(&mut self, _data: &mut SingleNumerovData<'_, P>, r_stop: f64) {
        self.take_per = (r_stop - self.r_min).abs() / (self.capacity as f64);
    }

    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        if data.r < self.r_min {
            return;
        }

        let append = self
            .distances
            .last()
            .is_none_or(|r| (r - data.r).abs() >= self.take_per);

        if append {
            if let Ok(s) = data.calculate_s_matrix() {
                self.distances.push(data.r);
                self.s_matrices.push(s)
            }
        }
    }
}

#[derive(Default)]
pub struct NumerovLogging<T> {
    r_min: f64,
    r_stop: f64,
    current: f64,
    steps_no: u64,
    _psi1: Option<T>,
}

impl<T> NumerovLogging<T> {
    pub fn steps_no(&self) -> u64 {
        self.steps_no
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for NumerovLogging<f64>
where
    P: SimplePotential,
{
    fn before(&mut self, data: &mut SingleNumerovData<'_, P>, r_stop: f64) {
        self.r_min = data.r;
        self.r_stop = r_stop;
        self.current = data.r;
    }

    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        self.current = data.r;
        self.steps_no += 1;
    }
}

impl<P> PropagatorModifier<MultiNumerovData<'_, P>> for NumerovLogging<Mat<f64>>
where
    P: MatPotential,
{
    fn before(&mut self, data: &mut MultiNumerovData<'_, P>, r_stop: f64) {
        self.r_min = data.r;
        self.r_stop = r_stop;
        self.current = data.r;
    }

    fn after_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        self.current = data.r;
        self.steps_no += 1;
    }
}

#[derive(Default)]
pub struct NumerovNodeCount {
    count: u64
}

impl NumerovNodeCount {
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for NumerovNodeCount
where
    P: SimplePotential,
{
    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        if data.psi1 < 0. {
            self.count += 1;
        }
    }
}

impl<P> PropagatorModifier<MultiNumerovData<'_, P>> for NumerovNodeCount
where
    P: MatPotential,
{
    fn after_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        if data.psi2_det < 0. {
            self.count += 1
        }
    }
}
