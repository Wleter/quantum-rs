use std::time::{Duration, Instant};

use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    observables::s_matrix::SMatrix,
    propagator::{Equation, Repr, Solution},
};

use super::Ratio;

pub trait PropagatorWatcher<T, R: Repr<T>> {
    fn before(&mut self, _sol: &Solution<R>, _eq: &Equation<T>, _r_stop: f64) {}

    fn after_step(&mut self, sol: &Solution<R>, eq: &Equation<T>);

    fn after_prop(&mut self, _sol: &Solution<R>, _eq: &Equation<T>) {}
}

pub struct ManyPropagatorWatcher<'a, T, R: Repr<T>> {
    modifiers: Vec<&'a mut dyn PropagatorWatcher<T, R>>,
}

impl<'a, T, R: Repr<T>> ManyPropagatorWatcher<'a, T, R> {
    pub fn new(modifiers: Vec<&'a mut dyn PropagatorWatcher<T, R>>) -> Self {
        Self { modifiers }
    }
}

impl<T, R: Repr<T>> PropagatorWatcher<T, R> for ManyPropagatorWatcher<'_, T, R> {
    fn before(&mut self, sol: &Solution<R>, eq: &Equation<T>, r_stop: f64) {
        for modifier in self.modifiers.iter_mut() {
            modifier.before(sol, eq, r_stop);
        }
    }

    fn after_step(&mut self, sol: &Solution<R>, eq: &Equation<T>) {
        for modifier in self.modifiers.iter_mut() {
            modifier.after_step(sol, eq);
        }
    }

    fn after_prop(&mut self, sol: &Solution<R>, eq: &Equation<T>) {
        for modifier in self.modifiers.iter_mut() {
            modifier.after_prop(sol, eq);
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
        let waves = Vec::with_capacity(capacity);

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

impl PropagatorWatcher<f64, Ratio<f64>> for WaveStorage<f64> {
    fn before(&mut self, sol: &Solution<Ratio<f64>>, _eq: &Equation<f64>, r_stop: f64) {
        if let SampleConfig::Step(value) = &mut self.sampling {
            *value = (sol.r - r_stop).abs() / self.capacity as f64
        }

        self.rs.push(sol.r);
        self.waves.push(self.last_value);
    }

    fn after_step(&mut self, sol: &Solution<Ratio<f64>>, _eq: &Equation<f64>) {
        self.last_value *= sol.sol.0;

        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(sol.r);
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
                if (sol.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(sol.r);
                    self.waves.push(self.last_value);
                }
            }
        }
    }
}

impl PropagatorWatcher<Mat<f64>, Ratio<Mat<f64>>> for WaveStorage<Mat<f64>> {
    fn before(&mut self, sol: &Solution<Ratio<Mat<f64>>>, _eq: &Equation<Mat<f64>>, r_stop: f64) {
        if let SampleConfig::Step(value) = &mut self.sampling {
            *value = (sol.r - r_stop).abs() / self.capacity as f64
        }

        self.rs.push(sol.r);
        self.waves.push(self.last_value.clone());
    }

    fn after_step(&mut self, sol: &Solution<Ratio<Mat<f64>>>, _eq: &Equation<Mat<f64>>) {
        self.last_value = &sol.sol.0 * &self.last_value;

        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(sol.r);
                    self.waves.push(self.last_value.clone());
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
                        .map(|(_, w)| w.clone())
                        .collect();
                }
            }
            SampleConfig::Step(sample_step) => {
                if (sol.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(sol.r);
                    self.waves.push(self.last_value.clone());
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

pub struct ScatteringVsDistance {
    pub(super) r_min: f64,
    pub(super) capacity: usize,
    pub(super) take_per: f64,

    pub distances: Vec<f64>,
    pub s_matrices: Vec<SMatrix>,
}

impl ScatteringVsDistance {
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

impl PropagatorWatcher<f64, Ratio<f64>> for ScatteringVsDistance {
    fn before(&mut self, sol: &Solution<Ratio<f64>>, _eq: &Equation<f64>, r_stop: f64) {
        assert!(sol.dr >= 0.);

        self.take_per = (r_stop - self.r_min).abs() / (self.capacity as f64);
    }

    fn after_step(&mut self, sol: &Solution<Ratio<f64>>, eq: &Equation<f64>) {
        if sol.r < self.r_min {
            return;
        }

        let append = self
            .distances
            .last()
            .is_none_or(|r| (r - sol.r).abs() >= self.take_per);

        if append {
            let s = sol.s_matrix(eq);
            self.distances.push(sol.r);
            self.s_matrices.push(s);
        }
    }
}

impl PropagatorWatcher<Mat<f64>, Ratio<Mat<f64>>> for ScatteringVsDistance {
    fn before(&mut self, sol: &Solution<Ratio<Mat<f64>>>, _eq: &Equation<Mat<f64>>, r_stop: f64) {
        assert!(sol.dr >= 0.);

        self.take_per = (r_stop - self.r_min).abs() / (self.capacity as f64);
    }

    fn after_step(&mut self, sol: &Solution<Ratio<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        if sol.r < self.r_min {
            return;
        }

        let append = self
            .distances
            .last()
            .is_none_or(|r| (r - sol.r).abs() >= self.take_per);

        if append {
            let s = sol.s_matrix(eq);
            self.distances.push(sol.r);
            self.s_matrices.push(s);
        }
    }
}

pub struct NumerovLogging {
    r_min: f64,
    r_stop: f64,
    current: f64,
    steps_no: u64,
    timer: Instant,
    progress: ProgressBar,
}

impl Default for NumerovLogging {
    fn default() -> Self {
        Self {
            r_min: Default::default(),
            r_stop: Default::default(),
            current: Default::default(),
            steps_no: Default::default(),
            timer: Instant::now(),
            progress: ProgressBar::hidden(),
        }
    }
}

impl NumerovLogging {
    pub fn steps_no(&self) -> u64 {
        self.steps_no
    }
}

impl<T, R: Repr<T>> PropagatorWatcher<T, R> for NumerovLogging {
    fn before(&mut self, sol: &Solution<R>, _eq: &Equation<T>, r_stop: f64) {
        self.r_min = sol.r;
        self.r_stop = r_stop;
        self.current = sol.r;

        self.timer = Instant::now();
        self.progress = ProgressBar::new((r_stop * 1000.0) as u64)
            .with_style(ProgressStyle::with_template("{bar:100.cyan/blue} {pos:>7} / {len:7} {msg}").unwrap())
            .with_message("mili bohr");
    }

    fn after_step(&mut self, sol: &Solution<R>, _eq: &Equation<T>) {
        self.progress.inc((1000.0 * (sol.r - self.current)) as u64);

        self.current = sol.r;
        self.steps_no += 1;
    }

    fn after_prop(&mut self, _sol: &Solution<R>, _eq: &Equation<T>) {
        self.progress.finish();

        println!("------------------------");
        println!("Numerov propagation done.\n");

        println!("Calculated in {} steps.\n", self.steps_no);

        let elapsed = self.timer.elapsed();
        let elapsed_step = Duration::from_secs_f64(elapsed.as_secs_f64() / self.steps_no as f64);

        println!("Calculated in time {}\n", elapsed.hhmmssxxx());
        println!("Mean time per step {}", elapsed_step.hhmmssxxx());
        println!("------------------------");
    }
}

// #[derive(Default)]
// pub struct NumerovNodeCount {
//     count: u64
// }

// impl NumerovNodeCount {
//     pub fn count(&self) -> u64 {
//         self.count
//     }
// }

// impl<P> PropagatorModifier<SingleNumerovData<'_, P>> for NumerovNodeCount
// where
//     P: SimplePotential,
// {
//     fn after_step(&mut self, data: &mut SingleNumerovData<'_, P>) {
//         if data.psi1 < 0. {
//             self.count += 1;
//         }
//     }
// }

// impl<P> PropagatorModifier<MultiNumerovData<'_, P>> for NumerovNodeCount
// where
//     P: MatPotential,
// {
//     fn after_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
//         if data.psi2_det < 0. {
//             self.count += 1
//         }
//     }
// }
