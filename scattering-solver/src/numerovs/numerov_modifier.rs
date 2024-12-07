use crate::{observables::s_matrix::{HasSMatrix, SingleSMatrix}, potentials::potential::SimplePotential};

use super::{propagator::PropagatorData, single_numerov::SingleNumerovData};

pub trait PropagatorModifier<D: PropagatorData> {
    fn before(&mut self, _data: &mut D, _r_stop: f64) {}

    fn after_step(&mut self, data: &mut D);

    fn after_prop(&mut self, _data: &mut D) {}
}

pub(super) enum SampleConfig {
    Each(usize),
    Step(f64)
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
            sampling
        }
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for WaveStorage<f64> 
where 
    P: SimplePotential
{
    fn before(&mut self, data: &mut SingleNumerovData<'_, P>, r_stop: f64) {
        match &mut self.sampling {
            SampleConfig::Step(value) => {
                *value = (data.r - r_stop).abs() / self.capacity as f64
            },
            _ => {},
        }

        self.rs.push(data.r);
    }

    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        self.last_value = data.psi1 * self.last_value;

        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value);
                }

                if self.rs.len() == self.capacity {
                    *sample_each *= 2;

                    self.rs = self.rs.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, r)| *r)
                        .collect();
    
                    self.waves = self.waves.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, w)| *w)
                        .collect();
                }

            },
            SampleConfig::Step(sample_step) => {
                if (data.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value);
                }
            },
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Sampling {
    Uniform,

    #[default]
    Variable,
}

pub struct ScatteringVsDistance<S: HasSMatrix> {
    r_min: f64,
    capacity: usize,
    take_per: f64,
    
    pub distances: Vec<f64>,
    pub s_matrices: Vec<S>
}

impl<S: HasSMatrix> ScatteringVsDistance<S> {
    pub fn new(r_min: f64, capacity: usize) -> Self {
        Self {
            r_min,
            capacity,
            take_per: 0.,
            distances: Vec::with_capacity(capacity),
            s_matrices: Vec::with_capacity(capacity)
        }
    }
}

impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for ScatteringVsDistance<SingleSMatrix>
where 
    P: SimplePotential
{
    fn before(&mut self, _data: &mut SingleNumerovData<'_, P>, r_stop: f64) {
        self.take_per = (r_stop - self.r_min).abs() / (self.capacity as f64);
    }

    fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
        if data.r < self.r_min {
            return;
        }

        let append = self.distances.last().map_or(true, |r| (r - data.r).abs() >= self.take_per);

        if append {
            if let Ok(s) = data.calculate_s_matrix() {
                self.distances.push(data.r);
                self.s_matrices.push(s)
            }
        }
    }
}
