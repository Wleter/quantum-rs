use std::time::{Duration, Instant};

use faer::{linalg::solvers::DenseSolveCore, Mat};
use hhmmss::Hhmmss;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    log_derivatives::LogDeriv,
    observables::s_matrix::SMatrix,
    propagator::{Equation, Repr, Solution},
};

use super::Ratio;

pub trait PropagatorWatcher<T, R: Repr<T>> {
    fn before(&mut self, _sol: &Solution<R>, _eq: &Equation<T>, _r_stop: f64) {}

    fn after_step(&mut self, sol: &Solution<R>, eq: &Equation<T>);

    fn after_prop(&mut self, _sol: &Solution<R>, _eq: &Equation<T>) {}
}

impl<T, R: Repr<T>, F: Fn(&Solution<R>, &Equation<T>)> PropagatorWatcher<T, R> for F {
    fn after_step(&mut self, sol: &Solution<R>, eq: &Equation<T>) {
        self(sol, eq)
    }
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
    pub nodes: Vec<u64>,

    pub(super) last_value: T,
    pub(super) counter: usize,
    pub(super) capacity: usize,
    pub(super) sampling: SampleConfig,
}

impl<T: Clone> WaveStorage<T> {
    pub fn new(sampling: Sampling, wave_init: T, capacity: usize) -> Self {
        let rs = Vec::with_capacity(capacity);
        let waves = Vec::with_capacity(capacity);
        let nodes = Vec::with_capacity(capacity);

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
            nodes,
        }
    }

    fn sample(&mut self, r: f64, nodes: u64) {
        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(r);
                    self.waves.push(self.last_value.clone());
                    self.nodes.push(nodes);
                }

                if self.rs.len() == self.capacity {
                    *sample_each *= 2;

                    halve_data(&self.rs);
                    halve_data(&self.waves);
                    halve_data(&self.nodes);
                }
            }
            SampleConfig::Step(sample_step) => {
                if (r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(r);
                    self.waves.push(self.last_value.clone());
                    self.nodes.push(nodes);
                }
            }
        }
    }

    fn before_internal(&mut self, r: f64, r_stop: f64) {
        if let SampleConfig::Step(value) = &mut self.sampling {
            *value = (r - r_stop).abs() / self.capacity as f64
        }

        self.rs.push(r);
        self.waves.push(self.last_value.clone());
        self.nodes.push(0);
    }
}

fn halve_data<T: Clone>(data: &[T]) -> Vec<T> {
    data.iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, w)| w.clone())
        .collect()
}

impl PropagatorWatcher<f64, Ratio<f64>> for WaveStorage<f64> {
    fn before(&mut self, sol: &Solution<Ratio<f64>>, _eq: &Equation<f64>, r_stop: f64) {
        self.before_internal(sol.r, r_stop);
    }

    fn after_step(&mut self, sol: &Solution<Ratio<f64>>, eq: &Equation<f64>) {
        let r_last = sol.r + sol.dr;
        let r_prev_last = sol.r;

        let f_last = 1. + sol.dr * sol.dr / 12. * eq.w_matrix(r_last);
        let f_prev_last = 1. + sol.dr * sol.dr / 12. * eq.w_matrix(r_prev_last);

        self.last_value *= sol.sol.0 * f_prev_last / f_last;

        self.sample(sol.r, sol.nodes);
    }
}

impl PropagatorWatcher<Mat<f64>, Ratio<Mat<f64>>> for WaveStorage<Mat<f64>> {
    fn before(&mut self, sol: &Solution<Ratio<Mat<f64>>>, _eq: &Equation<Mat<f64>>, r_stop: f64) {
        self.before_internal(sol.r, r_stop);
    }

    fn after_step(&mut self, sol: &Solution<Ratio<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        let size = eq.potential.size();
        let r_last = sol.r + sol.dr;
        let r_prev_last = sol.r;

        let mut f_last = Mat::zeros(size, size);
        eq.w_matrix(r_last, &mut f_last);
        f_last *= sol.dr * sol.dr / 12.;
        f_last += &eq.unit;

        let mut f_prev_last = Mat::zeros(size, size);
        eq.w_matrix(r_prev_last, &mut f_prev_last);
        f_prev_last *= sol.dr * sol.dr / 12.;
        f_prev_last += &eq.unit;

        self.last_value = f_last.partial_piv_lu().inverse() * &sol.sol.0 * f_prev_last * &self.last_value;

        self.sample(sol.r, sol.nodes);
    }
}

impl PropagatorWatcher<Mat<f64>, LogDeriv<Mat<f64>>> for WaveStorage<Mat<f64>> {
    fn before(
        &mut self,
        sol: &Solution<LogDeriv<Mat<f64>>>,
        _eq: &Equation<Mat<f64>>,
        r_stop: f64,
    ) {
        self.before_internal(sol.r, r_stop);
    }

    fn after_step(&mut self, sol: &Solution<LogDeriv<Mat<f64>>>, _eq: &Equation<Mat<f64>>) {
        self.last_value = &sol.sol.0 * &self.last_value * sol.dr + &self.last_value;

        self.sample(sol.r, sol.nodes);
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

pub struct PropagatorLogging {
    r_min: f64,
    r_stop: f64,
    current: f64,
    steps_no: u64,
    timer: Instant,
    progress: ProgressBar,
}

impl Default for PropagatorLogging {
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

impl PropagatorLogging {
    pub fn steps_no(&self) -> u64 {
        self.steps_no
    }
}

impl<T, R: Repr<T>> PropagatorWatcher<T, R> for PropagatorLogging {
    fn before(&mut self, sol: &Solution<R>, _eq: &Equation<T>, r_stop: f64) {
        self.r_min = sol.r;
        self.r_stop = r_stop;

        self.timer = Instant::now();
        self.progress = ProgressBar::new((r_stop * 1000.0) as u64)
            .with_style(
                ProgressStyle::with_template("{bar:100.cyan/blue} {pos:>7} / {len:7} {msg}")
                    .unwrap(),
            )
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
        if self.steps_no == 0 {
            return
        }
        let elapsed_step = Duration::from_secs_f64(elapsed.as_secs_f64() / self.steps_no as f64);

        println!("Calculated in time {}\n", elapsed.hhmmssxxx());
        println!("Mean time per step {}", elapsed_step.hhmmssxxx());
        println!("------------------------");
    }
}
