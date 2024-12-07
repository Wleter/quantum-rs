use std::f64::{consts::PI, INFINITY};

use num_traits::{One, Zero};
use quantum::{params::particles::Particles, units::{energy_units::Energy, Au}};

use crate::{
    boundary::{Boundary, Direction}, 
    numerovs::propagator::SamplingStorage, 
    potentials::potential::{Potential, SimplePotential}, 
};

use super::propagator::{MultiStep, Numerov, NumerovResult, Sampling, StepConfig};

/// Numerov method propagating ratios of the wave function,
/// implementing Numerov and NumerovResult trait for single channel and multi channel cases
pub struct RatioNumerov<'a, T, P>
where
    P: Potential<Space = T>,
{
    potential: &'a P,
    
    energy: f64,
    mass: f64,

    r: f64,
    dr: f64,
    psi1: T,
    psi2: T,

    f1: T,
    f2: T,
    f3: T,

    identity: T,
    current_g_func: T,

    current_potential: T,
    computation1: T,
    computation2: T,

    doubled_step_before: bool,
    is_set_up: bool,
    step_config: StepConfig,
}

impl<'a, T, P> RatioNumerov<'a, T, P>
where
    P: Potential<Space = T>,
{
    pub fn wave_last(&self) -> &T {
        &self.psi1
    }

    pub fn r(&self) -> f64 {
        self.r
    }

    pub fn dr(&self) -> f64 {
        self.dr
    }

    pub fn set_step_config(&mut self, config: StepConfig) {
        self.step_config = config;
    }
}

impl<'a, T, P> RatioNumerov<'a, T, P>
where
    T: Zero + One,
    P: Potential<Space = T>,
{
    /// Creates a new instance of the RatioNumerov struct
    pub fn new(particles: &Particles, potential: &'a P) -> Self {
        let mass = particles.red_mass();
        let energy = particles.get::<Energy<Au>>()
            .expect("Energy of the collision of type Energy<Au> not found")
            .value();

        Self {
            potential,
            energy,
            mass,

            r: 0.0,
            dr: 0.0,
            psi1: T::zero(),
            psi2: T::zero(),

            f1: T::zero(),
            f2: T::zero(),
            f3: T::zero(),

            identity: T::one(),
            current_g_func: T::zero(),
            current_potential: T::zero(),
            computation1: T::zero(),
            computation2: T::zero(),

            doubled_step_before: false,
            is_set_up: false,
            step_config: StepConfig::Variable(1.0, 0.0, INFINITY),
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// f64
/////////////////////////////////////////////////////////////////////////////////////////////////

impl<'a, P> RatioNumerov<'a, f64, P>
where
    P: Potential<Space = f64>,
{
    /// Returns the g function described in the Numerov method at position r
    fn g_func(&mut self, r: f64) -> f64 {
        2.0 * self.mass * (self.energy - self.potential.value(r))
    }

    pub(crate) fn propagate_node_counting(&mut self, r_stop: f64) -> usize {
        let mut node_count = 0;
        while self.r() < r_stop {
            self.single_step();

            if *self.wave_last() < 0.0 {
                node_count += 1;
            }
        }

        node_count
    }

    pub(crate) fn propagation_distance(&mut self, r_lims: (f64, f64)) -> f64 {
        let mut barrier = true;

        let mut decay_factor = 0.0;
        let max_decay = -(1e-5_f64.ln());

        let mut r = r_lims.0;
        self.current_g_func = self.g_func(r);
        let mut dr = self.local_step_size();
        
        while decay_factor < max_decay && r < r_lims.1 {
            get_step_size(&mut dr, self.local_step_size());
            r += dr;
            self.current_g_func = self.g_func(r);

            if self.current_g_func < 0.0 && !barrier {
                decay_factor += dr * self.current_g_func.abs().sqrt();
            } else if self.current_g_func >= 0.0 {
                barrier = false;
            }
        }

        if barrier {
            println!("Energy below/near potential minimum, possible long computations todo!")
        }

        r.min(r_lims.1)
    }

    pub(crate) fn potential_minimum(&mut self, r_lims: (f64, f64)) -> f64 {
        let mut r = r_lims.0;

        let mut potential_minimum = self.potential.value(r);
        self.current_g_func = self.g_func(r);
        let mut dr = self.local_step_size();
        
        while r < r_lims.1 {
            get_step_size(&mut dr, self.local_step_size());

            r += dr;
            self.current_g_func = self.g_func(r);

            let potential = self.potential.value(r);
            if potential < potential_minimum {
                potential_minimum = potential;
            }
        }

        potential_minimum
    }
}

fn get_step_size(curr_dr: &mut f64, recommended_step: f64) {
    if recommended_step > 2.0 * curr_dr.abs() {
        *curr_dr *= 2.0;
    } else {
        while 1.2 * recommended_step < curr_dr.abs() {
            *curr_dr /= 2.0;
        }
    }
}

impl<'a, P> MultiStep<P> for RatioNumerov<'a, f64, P>
where
    P: Potential<Space = f64>,
{
    fn variable_step(&mut self) {
        self.current_g_func = self.g_func(self.r + self.dr);

        let step_size = self.local_step_size();
        if step_size > 2.0 * self.dr.abs() && !self.doubled_step_before {
            self.doubled_step_before = true;
            self.double_step();
            self.current_g_func = self.g_func(self.r + self.dr);
        } else {
            self.doubled_step_before = false;
            let mut halved = false;
            while 1.2 * step_size < self.dr.abs() {
                self.halve_step();
                halved = true;
            }

            if halved {
                self.current_g_func = self.g_func(self.r + self.dr);
            }
        }

        self.step();
    }

    fn step(&mut self) {
        self.r += self.dr;

        let f = 1.0 + self.dr * self.dr * self.current_g_func / 12.0;
        let psi = (12.0 - 10.0 * self.f1 - self.f2 / self.psi1) / f;

        self.f3 = self.f2;
        self.f2 = self.f1;
        self.f1 = f;

        self.psi2 = self.psi1;
        self.psi1 = psi;
    }

    fn halve_step(&mut self) {
        self.dr /= 2.0;

        let f = 1.0 + self.dr * self.dr * self.g_func(self.r - self.dr) / 12.0;
        self.f1 = self.f1 / 4.0 + 0.75;
        self.f2 = self.f2 / 4.0 + 0.75;

        let psi = (self.f1 * self.psi1 + self.f2) / (12.0 - 10.0 * f);
        self.f2 = f;

        self.psi2 = psi;
        self.psi1 /= psi;
    }

    fn double_step(&mut self) {
        self.dr *= 2.0;

        self.f2 = 4.0 * self.f3 - 3.0;
        self.f1 = 4.0 * self.f1 - 3.0;

        self.psi1 *= self.psi2;
    }

    fn local_step_size(&self) -> f64 {
        match self.step_config {
            StepConfig::Fixed(dr) => dr,
            StepConfig::Variable(step_factor, min_value, max_value) => {
                let lambda = 2.0 * PI / self.current_g_func.abs().sqrt();
                let lambda_step_ratio = 500.0;
        
                (step_factor * lambda / lambda_step_ratio).clamp(min_value, max_value)
            }
        }
    }
}

impl<'a, P> Numerov<f64, P> for RatioNumerov<'a, f64, P>
where
    P: Potential<Space = f64>,
{
    fn prepare(&mut self, boundary: &Boundary<f64>) {
        self.r = boundary.r_start;
        self.current_g_func = self.g_func(boundary.r_start);

        self.dr = match boundary.direction {
            Direction::Inwards => -self.local_step_size(),
            Direction::Outwards => self.local_step_size(),
            Direction::Step(dr) => dr,
        };

        self.psi1 = boundary.start_value;
        self.psi2 = boundary.before_value;

        self.f3 = 1.0 + self.dr * self.dr * self.g_func(self.r - 2.0 * self.dr) / 12.0;
        self.f2 = 1.0 + self.dr * self.dr * self.g_func(self.r - self.dr) / 12.0;
        self.f1 = 1.0 + self.dr * self.dr * self.current_g_func / 12.0;

        self.is_set_up = true;
    }
    
    fn single_step(&mut self) {
        assert!(self.is_set_up, "Numerov method not set up");
        self.variable_step();
    }

    fn propagate_to(&mut self, r: f64) {
        while self.dr.signum() * (r - self.r) > 0.0 {
            self.variable_step();
        }
    }

    fn propagate_values(&mut self, r_stop: f64, wave_init: f64, sampling: Sampling, capacity: usize) -> (Vec<f64>, Vec<f64>) {
        assert!(self.is_set_up, "Numerov method not set up");
        let mut sampler = SamplingStorage::new(sampling, self.r, &wave_init, r_stop, capacity);

        let mut psi_actual = wave_init;
        while self.dr.signum() * (r_stop - self.r) > 0.0 {
            self.variable_step();

            psi_actual *= self.psi1;

            sampler.sample(self.r, &psi_actual);
        }

        sampler.result()
    }

    fn result(&self) -> NumerovResult<f64> {
        NumerovResult {
            r_last: self.r,
            dr: self.dr,
            wave_ratio: self.psi1,
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// SMatrix<f64, N, N>
/////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "nalgebra")]
use nalgebra::SMatrix;

#[cfg(feature = "nalgebra")]
impl<'a, const N: usize, P> RatioNumerov<'a, SMatrix<f64, N, N>, P>
where
    P: Potential<Space = SMatrix<f64, N, N>>,
{
    /// Returns the g function described in the Numerov method at position r
    fn g_func(&mut self, r: f64) {
        self.potential.value_inplace(r, &mut self.current_potential);

        self.current_g_func = - 2. * self.mass * self.current_potential;
        self.current_g_func += 2.0 * self.mass * self.energy * self.identity;
    }
}

impl<'a, const N: usize, P> MultiStep<P> for RatioNumerov<'a, SMatrix<f64, N, N>, P>
where
    P: Potential<Space = SMatrix<f64, N, N>>,
{
    fn variable_step(&mut self) {
        self.g_func(self.r + self.dr);

        let step_size = self.local_step_size();
        if step_size > 2.0 * self.dr.abs() && !self.doubled_step_before {
            self.doubled_step_before = true;
            self.double_step();

            self.g_func(self.r + self.dr);
        } else {
            self.doubled_step_before = false;
            let mut halved = false;
            while 1.2 * step_size < self.dr.abs() {
                halved = true;
                self.halve_step();
            }
            if halved {
                self.g_func(self.r + self.dr);
            }
        }

        self.step();
    }

    fn step(&mut self) {
        self.r += self.dr;

        self.computation1 = self.dr * self.dr * self.current_g_func / 12.0;
        self.computation1 += self.identity; // hold f value in self.computation1

        self.f3 = self.computation1;  // hold next computations of f in self.f3

        if !self.f3.try_inverse_mut() {
            panic!("Inversion of f matrix failed");
        }

        self.computation2 = 12.0 * self.identity;
        self.computation1.gemm();

        let psi = self.f3 * (12.0 * self.identity - 10.0 * self.f1 - self.f2 * self.psi1.try_inverse().unwrap());

        self.f3 = self.f2;
        self.f2 = self.f1;
        self.f1 = f;

        self.psi2 = self.psi1;
        self.psi1 = psi;
    }

    fn halve_step(&mut self) {
        self.dr /= 2.0;

        let f = self.identity + self.dr * self.dr * self.g_func(self.r - self.dr) / 12.0;
        self.f1 = self.f1 / 4.0 + 0.75 * self.identity;
        self.f2 = self.f2 / 4.0 + 0.75 * self.identity;

        let psi = (12.0 * self.identity - 10.0 * f).try_inverse().unwrap()
            * (self.f1 * self.psi1 + self.f2);
        self.f2 = f;

        self.psi2 = psi;
        self.psi1 *= psi.try_inverse().unwrap();
    }

    fn double_step(&mut self) {
        self.dr *= 2.0;

        self.f2 = 4.0 * self.f3 - 3.0 * self.identity;
        self.f1 = 4.0 * self.f1 - 3.0 * self.identity;

        self.psi1 *= self.psi2;
    }

    fn local_step_size(&self) -> f64 {
        match self.step_config {
            StepConfig::Fixed(dr) => dr,
            StepConfig::Variable(step_factor, min_value, max_value) => {
                let max_g_func_val = self
                    .current_g_func
                    .iter()
                    .fold(0.0, |acc, &x| x.abs().max(acc));
        
                let lambda = 2.0 * PI / max_g_func_val.sqrt();
                let lambda_step_ratio = 500.0;
        
                (step_factor * lambda / lambda_step_ratio).clamp(min_value, max_value)
            }
        }
    }
}

impl<'a, const N: usize, P> Numerov<FMatrix<N>, P> for RatioNumerov<'a, FMatrix<N>, P>
where
    P: Potential<Space = FMatrix<N>>,
{
    fn prepare(&mut self, boundary: &Boundary<FMatrix<N>>) {
        self.r = boundary.r_start;

        self.current_g_func = self.g_func(&boundary.r_start);
        self.dr = match boundary.direction {
            Direction::Inwards => -self.recommended_step_size(),
            Direction::Outwards => self.recommended_step_size(),
            Direction::Step(dr) => dr,
        };

        self.psi1 = boundary.start_value;
        self.psi2 = boundary.before_value;

        self.f3 = self.identity + self.dr * self.dr * self.g_func(&(self.r - 2.0 * self.dr)) / 12.0;
        self.f2 = self.identity + self.dr * self.dr * self.g_func(&(self.r - self.dr)) / 12.0;
        self.f1 = self.identity + self.dr * self.dr * self.current_g_func / 12.0;

        self.is_set_up = true;
    }
        
    fn single_step(&mut self) {
        assert!(self.is_set_up, "Numerov method not set up");
        self.variable_step();
    }

    fn propagate_to(&mut self, r: f64) {
        while self.dr.signum() * (r - self.r) > 0.0 {
            self.variable_step();
        }
    }

    fn propagate_values(&mut self, r_stop: f64, wave_init: FMatrix<N>, sampling: Sampling) -> (Vec<f64>, Vec<FMatrix<N>>) {
        assert!(self.is_set_up, "Numerov method not set up");
        let mut sampler = SamplingStorage::new(sampling, &self.r, &wave_init, &r_stop);

        let mut psi_actual = wave_init;

        while self.dr.signum() * (r_stop - self.r) > 0.0 {
            self.variable_step();

            psi_actual = self.psi1 * psi_actual;

            sampler.sample(&self.r, &psi_actual);
        }

        sampler.result()
    }

    fn result(&self) -> NumerovResult<FMatrix<N>> {
        NumerovResult {
            r_last: self.r,
            dr: self.dr,
            wave_ratio: self.psi1,
        }
    }
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// // CMatrix<N>
// /////////////////////////////////////////////////////////////////////////////////////////////////

// impl<'a, const N: usize, P> RatioNumerov<'a, CMatrix<N>, P>
// where
//     P: Potential<Space = CMatrix<N>>,
// {
//     /// Returns the g function described in the Numerov method at position r
//     fn g_func(&self, &r: &f64) -> CMatrix<N> {
//         (self.identity * Complex64::from(self.energy) - self.collision_params.potential.value(&r))
//             * Complex64::from(2.0 * self.mass)
//     }
// }

// impl<'a, const N: usize, P> MultiStep<P> for RatioNumerov<'a, CMatrix<N>, P>
// where
//     P: Potential<Space = CMatrix<N>>,
// {
//     fn variable_step(&mut self) {
//         self.current_g_func = self.g_func(&(self.r + self.dr));

//         let step_size = self.recommended_step_size();

//         if step_size > 2.0 * self.dr.abs() && !self.doubled_step_before {
//             self.doubled_step_before = true;
//             self.double_step();
//             self.current_g_func = self.g_func(&(self.r + self.dr));
//         } else {
//             self.doubled_step_before = false;
//             let mut halved = false;
//             while 1.2 * step_size < self.dr.abs() {
//                 halved = true;
//                 self.half_step();
//             }
//             if halved {
//                 self.current_g_func = self.g_func(&(self.r + self.dr));
//             }
//         }

//         self.step();
//     }

//     fn step(&mut self) {
//         self.r += self.dr;

//         let f = self.identity + self.current_g_func * Complex64::from(self.dr * self.dr / 12.0);
//         let psi = f.try_inverse().unwrap()
//             * (self.identity * Complex64::from(12.0)
//                 - self.f1 * Complex64::from(10.0)
//                 - self.f2 * self.psi1.try_inverse().unwrap());

//         self.f3 = self.f2;
//         self.f2 = self.f1;
//         self.f1 = f;

//         self.psi2 = self.psi1;
//         self.psi1 = psi;
//     }

//     fn half_step(&mut self) {
//         self.dr /= 2.0;

//         let f = self.identity
//             + self.g_func(&(self.r - self.dr)) * Complex64::from(self.dr * self.dr / 12.0);
//         self.f1 = self.f1 / Complex64::from(4.0) + self.identity * Complex64::from(0.75);
//         self.f2 = self.f2 / Complex64::from(4.0) + self.identity * Complex64::from(0.75);

//         let psi = (self.identity * Complex64::from(12.0) - f * Complex64::from(10.0))
//             .try_inverse()
//             .unwrap()
//             * (self.f1 * self.psi1 + self.f2);
//         self.f2 = f;

//         self.psi2 = psi;
//         self.psi1 *= psi.try_inverse().unwrap();
//     }

//     fn double_step(&mut self) {
//         self.dr *= 2.0;

//         self.f2 = self.f3 * Complex64::from(4.0) - self.identity * Complex64::from(3.0);
//         self.f1 = self.f1 * Complex64::from(4.0) - self.identity * Complex64::from(3.0);

//         self.psi1 *= self.psi2;
//     }

//     fn recommended_step_size(&self) -> f64 {
//         match self.step_config {
//             StepConfig::Fixed(dr) => dr,
//             StepConfig::Variable(step_factor, min_value, max_value) => {
//                 let max_g_func_val = self
//                     .current_g_func
//                     .iter()
//                     .fold(0.0, |acc, &x| x.norm().max(acc));
    
//                 let lambda = 2.0 * PI / max_g_func_val.sqrt();
//                 let lambda_step_ratio = 500.0;
        
//                 (step_factor * lambda / lambda_step_ratio).clamp(min_value, max_value)
//             }
//         }
//     }
// }

// impl<'a, const N: usize, P> Numerov<CMatrix<N>, P> for RatioNumerov<'a, CMatrix<N>, P>
// where
//     P: Potential<Space = CMatrix<N>>,
// {
//     fn prepare(&mut self, boundary: &Boundary<CMatrix<N>>) {
//         self.r = boundary.r_start;

//         self.current_g_func = self.g_func(&boundary.r_start);
//         self.dr = self.recommended_step_size();
//         self.dr = match boundary.direction {
//             Direction::Inwards => -self.recommended_step_size(),
//             Direction::Outwards => self.recommended_step_size(),
//             Direction::Step(dr) => dr,
//         };

//         self.psi1 = boundary.start_value;
//         self.psi2 = boundary.before_value;

//         self.f3 = self.identity
//             + self.g_func(&(self.r - 2.0 * self.dr)) * Complex64::from(self.dr * self.dr / 12.0);
//         self.f2 = self.identity
//             + self.g_func(&(self.r - self.dr)) * Complex64::from(self.dr * self.dr / 12.0);
//         self.f1 = self.identity + self.current_g_func * Complex64::from(self.dr * self.dr / 12.0);

//         self.is_set_up = true;
//     }
        
//     fn single_step(&mut self) {
//         assert!(self.is_set_up, "Numerov method not set up");
//         self.variable_step();
//     }

//     fn propagate_to(&mut self, r: f64) {
//         while self.dr.signum() * (r - self.r) > 0.0 {
//             self.variable_step();
//         }
//     }

//     fn propagate_values(&mut self, r: f64, wave_init: CMatrix<N>, sampling: Sampling) -> (Vec<f64>, Vec<CMatrix<N>>) {
//         assert!(self.is_set_up, "Numerov method not set up");
//         let mut sampler = SamplingStorage::new(sampling, &self.r, &wave_init, &r);

//         let mut psi_actual = wave_init;

//         while self.dr.signum() * (r - self.r) > 0.0 {
//             self.variable_step();

//             psi_actual = self.psi1 * psi_actual;

//             sampler.sample(&self.r, &psi_actual);
//         }

//         sampler.result()
//     }

//     fn result(&self) -> NumerovResult<CMatrix<N>> {
//         NumerovResult {
//             r_last: self.r,
//             dr: self.dr,
//             wave_ratio: self.psi1,
//         }
//     }
// }

// /////////////////////////////////////////////////////////////////////////////////////////////////
// // DFMatrix
// /////////////////////////////////////////////////////////////////////////////////////////////////

// impl<'a, P> RatioNumerov<'a, DFMatrix, P>
// where
//     P: Potential<Space = DFMatrix>,
// {
//     /// Creates a new instance of the RatioNumerov struct
//     pub fn new_dyn(collision_params: &'a CollisionParams<P>) -> Self {
//         let mass = collision_params.particles.red_mass();
//         let energy = collision_params.particles.internals.get_value("energy");

//         let asymptotic = collision_params.potential.asymptotic_value();
//         let size = asymptotic.nrows();
//         assert!(size == asymptotic.ncols(), "Potential matrix must be square");

//         Self {
//             collision_params,
//             energy,
//             mass,

//             r: 0.0,
//             dr: 0.0,
//             psi1: DFMatrix::zeros(size, size),
//             psi2: DFMatrix::zeros(size, size),

//             f1: DFMatrix::zeros(size, size),
//             f2: DFMatrix::zeros(size, size),
//             f3: DFMatrix::zeros(size, size),

//             identity: DFMatrix::identity(size, size),
//             current_g_func: DFMatrix::zeros(size, size),

//             doubled_step_before: false,
//             is_set_up: false,
//             step_config: StepConfig::Variable(1.0, 0.0, INFINITY),
//         }
//     }
// }

// impl<'a, P> RatioNumerov<'a, DFMatrix, P>
// where
//     P: Potential<Space = DFMatrix>,
// {
//     /// Returns the g function described in the Numerov method at position r
//     fn g_func(&self, &r: &f64) -> DFMatrix {
//         2.0 * self.mass * (self.energy * &self.identity - self.collision_params.potential.value(&r))
//     }

//     pub(crate) fn propagate_node_counting(&mut self, r_stop: f64) -> usize {
//         let mut node_count = 0;
//         while self.r() < r_stop {
//             self.single_step();

//             if self.wave_last().determinant() < 0.0 {
//                 node_count += 1;
//             }
//         }

//         node_count
//     }

//     pub(crate) fn potential_minimum(&mut self, r_lims: (f64, f64)) -> f64 {
//         let mut r = r_lims.0;

//         let mut potential_minimum = self.collision_params.potential.value(&r)
//             .symmetric_eigenvalues()
//             .iter()
//             .min_by(|&x, &y| x.partial_cmp(y).unwrap())
//             .unwrap()
//             .to_owned();

//         self.current_g_func = self.g_func(&r);
//         let mut dr = self.recommended_step_size();
        
//         while r < r_lims.1 {
//             get_step_size(&mut dr, self.recommended_step_size());

//             r += dr;
//             self.current_g_func = self.g_func(&r);

//             let potential = self.collision_params.potential.value(&r)
//                 .symmetric_eigenvalues()
//                 .iter()
//                 .min_by(|&x, &y| x.partial_cmp(y).unwrap())
//                 .unwrap()
//                 .to_owned();

//             if potential < potential_minimum {
//                 potential_minimum = potential;
//             }
//         }

//         potential_minimum
//     }
// }

// impl<'a, P> MultiStep<P> for RatioNumerov<'a, DFMatrix, P>
// where
//     P: Potential<Space = DFMatrix>,
// {
//     fn variable_step(&mut self) {
//         self.current_g_func = self.g_func(&(self.r + self.dr));

//         let step_size = self.recommended_step_size();
//         if step_size > 2.0 * self.dr.abs() && !self.doubled_step_before {
//             self.doubled_step_before = true;
//             self.double_step();
//             self.current_g_func = self.g_func(&(self.r + self.dr));
//         } else {
//             self.doubled_step_before = false;
//             let mut halved = false;
//             while 1.2 * step_size < self.dr.abs() {
//                 halved = true;
//                 self.half_step();
//             }
//             if halved {
//                 self.current_g_func = self.g_func(&(self.r + self.dr));
//             }
//         }

//         self.step();
//     }

//     fn step(&mut self) {
//         self.r += self.dr;

//         let f = &self.identity + self.dr * self.dr * &self.current_g_func / 12.0;
//         let psi = f.clone().try_inverse().unwrap()
//             * (12.0 * &self.identity - 10.0 * &self.f1 - &self.f2 * self.psi1.clone().try_inverse().unwrap());

//         swap(&mut self.f3, &mut self.f2);
//         swap(&mut self.f2, &mut self.f1);
//         self.f1 = f;

//         swap(&mut self.psi2, &mut self.psi1);
//         self.psi1 = psi;
//     }

//     fn half_step(&mut self) {
//         self.dr /= 2.0;

//         let f = &self.identity + self.dr * self.dr * self.g_func(&(self.r - self.dr)) / 12.0;
//         self.f1 = &self.f1 / 4.0 + 0.75 * &self.identity;
//         self.f2 = &self.f2 / 4.0 + 0.75 * &self.identity;

//         let psi = (12.0 * &self.identity - 10.0 * &f).try_inverse().unwrap()
//             * (&self.f1 * &self.psi1 + &self.f2);
//         self.f2 = f;

//         self.psi2 = psi.clone();
//         self.psi1 *= psi.try_inverse().unwrap();
//     }

//     fn double_step(&mut self) {
//         self.dr *= 2.0;

//         self.f2 = 4.0 * &self.f3 - 3.0 * &self.identity;
//         self.f1 = 4.0 * &self.f1 - 3.0 * &self.identity;

//         self.psi1 *= &self.psi2;
//     }

//     fn recommended_step_size(&self) -> f64 {
//         match self.step_config {
//             StepConfig::Fixed(dr) => dr,
//             StepConfig::Variable(step_factor, min_value, max_value) => {
//                 let max_g_func_val = self
//                     .current_g_func
//                     .iter()
//                     .fold(0.0, |acc, &x| x.abs().max(acc));
    
//                 let lambda = 2.0 * PI / max_g_func_val.sqrt();
//                 let lambda_step_ratio = 500.0;
        
//                 (step_factor * lambda / lambda_step_ratio).clamp(min_value, max_value)
//             }
//         }
//     }
// }

// impl<'a, P> Numerov<DFMatrix, P> for RatioNumerov<'a, DFMatrix, P>
// where
//     P: Potential<Space = DFMatrix>,
// {
//     fn prepare(&mut self, boundary: &Boundary<DFMatrix>) {
//         self.r = boundary.r_start;

//         self.current_g_func = self.g_func(&boundary.r_start);
//         self.dr = match boundary.direction {
//             Direction::Inwards => -self.recommended_step_size(),
//             Direction::Outwards => self.recommended_step_size(),
//             Direction::Step(dr) => dr,
//         };

//         self.psi1 = boundary.start_value.clone();
//         self.psi2 = boundary.before_value.clone();

//         self.f3 = &self.identity + self.dr * self.dr * &self.g_func(&(self.r - 2.0 * self.dr)) / 12.0;
//         self.f2 = &self.identity + self.dr * self.dr * &self.g_func(&(self.r - self.dr)) / 12.0;
//         self.f1 = &self.identity + self.dr * self.dr * &self.current_g_func / 12.0;

//         self.is_set_up = true;
//     }
        
//     fn single_step(&mut self) {
//         assert!(self.is_set_up, "Numerov method not set up");
//         self.variable_step();
//     }

//     fn propagate_to(&mut self, r: f64) {
//         while self.dr.signum() * (r - self.r) > 0.0 {
//             self.variable_step();
//         }
//     }

//     fn propagate_values(&mut self, r_stop: f64, wave_init: DFMatrix, sampling: Sampling) -> (Vec<f64>, Vec<DFMatrix>) {
//         assert!(self.is_set_up, "Numerov method not set up");
//         let mut sampler = SamplingStorage::new(sampling, &self.r, &wave_init, &r_stop);

//         let mut psi_actual = wave_init;

//         while self.dr.signum() * (r_stop - self.r) > 0.0 {
//             self.variable_step();

//             psi_actual = &self.psi1 * psi_actual;

//             sampler.sample(&self.r, &psi_actual);
//         }

//         sampler.result()
//     }

//     fn result(&self) -> NumerovResult<DFMatrix> {
//         NumerovResult {
//             r_last: self.r,
//             dr: self.dr,
//             wave_ratio: self.psi1.clone(),
//         }
//     }
// }