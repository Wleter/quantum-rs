use core::f64;
use std::{f64::consts::PI, mem::swap};

use nalgebra::DMatrix;
use num::complex::Complex64;
use quantum::{params::particles::Particles, units::{energy_units::Energy, mass_units::Mass, Au}, utility::{asymptotic_bessel_j, asymptotic_bessel_n, bessel_j_ratio, bessel_n_ratio}};
use crate::{boundary::Boundary, numerovs::{numerov_modifier::{PropagatorModifier, SampleConfig, WaveStorage}, propagator::{MultiStep, MultiStepRule, Numerov, NumerovResult, PropagatorData, StepAction, StepRule}}, observables::s_matrix::nalgebra::NalgebraSMatrix, potentials::{dispersion_potential::Dispersion, potential::{Dimension, Potential, SimplePotential}, potential_factory::create_centrifugal}, utility::AngularSpin};
use super::MultiRatioNumerovStep;

#[derive(Clone)]
pub struct MultiNumerovDataDMatrix<'a, P>
where 
    P: Potential<Space = DMatrix<f64>>
{
    pub(super) r: f64,
    pub(super) dr: f64,

    pub(super) potential: &'a P,
    pub(super) centrifugal: Option<Dispersion>,
    pub(super) mass: f64,
    pub(super) energy: f64,
    pub(super) l: AngularSpin,

    pub(super) potential_buffer: DMatrix<f64>,
    pub(super) unit: DMatrix<f64>,
    pub(super) current_g_func: DMatrix<f64>,

    pub(super) psi1: DMatrix<f64>,
    pub(super) psi2: DMatrix<f64>,
}


pub type DMatrixRatioNumerov<'a, P> = Numerov<
    MultiNumerovDataDMatrix<'a, P>, 
    MultiStepRule<MultiNumerovDataDMatrix<'a, P>>, 
    MultiRatioNumerovStep<DMatrix<f64>>
>;

impl<P> MultiNumerovDataDMatrix<'_, P> 
where 
    P: Potential<Space = DMatrix<f64>>
{
    pub fn get_g_func(&mut self, r: f64, out: &mut DMatrix<f64>) {
        self.potential.value_inplace(r, &mut self.potential_buffer);


        if let Some(centr) = &self.centrifugal {
            let centr = centr.value(r);

            for i in 0..self.potential_buffer.nrows() {
                unsafe {
                    *self.potential_buffer.get_unchecked_mut((i, i)) += centr
                }
            }
        }

        out.zip_zip_apply(&self.unit, &self.potential_buffer, |o, u, p| {
            *o = 2.0 * self.mass * (self.energy * u - p)
        });
    }

    pub fn calculate_s_matrix(&self, entrance: usize) -> NalgebraSMatrix {
        let size = self.potential.size();
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;
        let wave_ratio = &self.psi1;

        let mut asymptotic = DMatrix::zeros(size, size);
        self.potential.value_inplace(r_last, &mut asymptotic);

        // todo! assume diagonality of the asymptotic potential

        let is_open_channel = asymptotic
            .diagonal()
            .iter()
            .map(|&val| val < self.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .diagonal()
            .iter()
            .map(|&val| (2.0 * self.mass * (self.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = DMatrix::zeros(size, size);
        let mut j_prev_last = DMatrix::zeros(size, size);
        let mut n_last = DMatrix::zeros(size, size);
        let mut n_prev_last = DMatrix::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            if is_open_channel[i] {
                j_last[(i, i)] = asymptotic_bessel_j(momentum * r_last, self.l.0);
                j_prev_last[(i, i)] = asymptotic_bessel_j(momentum * r_prev_last, self.l.0);
                n_last[(i, i)] = asymptotic_bessel_n(momentum * r_last, self.l.0);
                n_prev_last[(i, i)] = asymptotic_bessel_n(momentum * r_prev_last, self.l.0);
            } else {
                j_last[(i, i)] = bessel_j_ratio(momentum * r_last, momentum * r_prev_last);
                j_prev_last[(i, i)] = 1.0;
                n_last[(i, i)] = bessel_n_ratio(momentum * r_last, momentum * r_prev_last);
                n_prev_last[(i, i)] = 1.0;
            }
        }

        let denominator = (wave_ratio * n_prev_last - n_last).try_inverse().unwrap();

        let k_matrix = -denominator * (wave_ratio * j_prev_last - j_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = DMatrix::<Complex64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while !is_open_channel[i_full] {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while !is_open_channel[j_full] {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = Complex64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = DMatrix::<Complex64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).try_inverse().unwrap();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel.iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == entrance)
            .expect("Closed entrance channel")
            .0;

        NalgebraSMatrix::new(s_matrix, momenta, entrance)
    }
}

impl<'a, P, S, M> Numerov<MultiNumerovDataDMatrix<'a, P>, S, M> 
where 
    P: Potential<Space = DMatrix<f64>>,
    S: StepRule<MultiNumerovDataDMatrix<'a, P>>,
    M: MultiStep<MultiNumerovDataDMatrix<'a, P>>
{
    pub fn get_result(&self) -> NumerovResult<DMatrix<f64>> {
        NumerovResult {
            r_last: self.data.r,
            dr: self.data.dr,
            wave_ratio: self.data.psi1.clone(),
        }
    }
}

impl<'a, P, S> Numerov<MultiNumerovDataDMatrix<'a, P>, S, MultiRatioNumerovStep<DMatrix<f64>>>
where 
    P: Potential<Space = DMatrix<f64>>,
    S: StepRule<MultiNumerovDataDMatrix<'a, P>>,
{
    pub fn new(potential: &'a P, particles: &Particles, step_rules: S, boundary: Boundary<DMatrix<f64>>) -> Self {
        let mass = particles.get::<Mass<Au>>()
            .expect("no reduced mass parameter Mass<Au> found in particles")
            .to_au();
        let energy = particles.get::<Energy<Au>>()
            .expect("no collision energy Energy<Au> found in particles")
            .to_au();

        let l = *particles.get::<AngularSpin>().unwrap_or(&AngularSpin(0));
        let centrifugal = create_centrifugal(mass, l);

        let size = potential.size();

        let r = boundary.r_start;
        let mut data = MultiNumerovDataDMatrix {
            r,
            dr: 0.,
            potential,
            centrifugal,
            l,
            mass,
            energy,
            potential_buffer: DMatrix::zeros(size, size),
            unit: DMatrix::identity(size, size),
            current_g_func: DMatrix::zeros(size, size),
            psi1: DMatrix::zeros(size, size),
            psi2: DMatrix::zeros(size, size),
        };

        data.current_g_func();
    
        let dr = match boundary.direction {
            crate::boundary::Direction::Inwards => -step_rules.get_step(&data),
            crate::boundary::Direction::Outwards => step_rules.get_step(&data),
            crate::boundary::Direction::Step(dr) => dr,
        };
        data.dr = dr;
    
        data.psi1 = boundary.start_value;
        data.psi2 = boundary.before_value;
    
        let mut f3 = DMatrix::zeros(size, size);
        data.get_g_func(r - 2. * dr, &mut f3);

        let mut f2 = DMatrix::zeros(size, size);
        data.get_g_func(r - dr, &mut f2);

        let f3 = &data.unit + dr * dr / 12. * f3;
        let f2 = &data.unit + dr * dr / 12. * f2;
        let f1 = &data.unit + dr * dr / 12. * &data.current_g_func;

        let multi_step = MultiRatioNumerovStep { 
            f1, 
            f2, 
            f3, 
            buffer1: DMatrix::zeros(size, size), 
            buffer2: DMatrix::zeros(size, size)
        };

        Self {
            data,
            step_rules,
            multi_step,
        }
    }
}

impl<P> PropagatorData for MultiNumerovDataDMatrix<'_, P> 
where 
    P: Potential<Space = DMatrix<f64>>
{
    fn step_size(&self) -> f64 {
        self.dr
    }
    
    fn current_g_func(&mut self) {
        self.potential.value_inplace(self.r + self.dr, &mut self.potential_buffer);

        if let Some(centr) = &self.centrifugal {
            let centr = centr.value(self.r + self.dr);

            for i in 0..self.potential_buffer.nrows() {
                unsafe {
                    *self.potential_buffer.get_unchecked_mut((i, i)) += centr
                }
            }
        }

        self.current_g_func.zip_zip_apply(&self.unit, &self.potential_buffer, |o, u, p| {
            *o = 2.0 * self.mass * (self.energy * u - p)
        });
    }

    fn advance(&mut self) {
        self.r += self.dr;
    }

    fn crossed_distance(&self, r: f64) -> bool {
        self.dr.signum() * (r - self.r) <= 0.0
    }
}

impl<P> StepRule<MultiNumerovDataDMatrix<'_, P>> for MultiStepRule<MultiNumerovDataDMatrix<'_, P>>
where 
    P: Potential<Space = DMatrix<f64>>
{
    fn get_step(&self, data: &MultiNumerovDataDMatrix<P>) -> f64 {
        let mut max_g_val = f64::MIN;
        for i in 0..data.current_g_func.size() {
            unsafe {
                max_g_val = f64::max(max_g_val, *data.current_g_func.get_unchecked((i, i)))
            }
        }

        let lambda = 2. * PI / max_g_val.abs().sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }
    
    fn assign(&mut self, data: &MultiNumerovDataDMatrix<P>) -> StepAction {
        let prop_step = data.step_size();
        let step = self.get_step(data);

        if prop_step > 1.2 * step {
            self.doubled_step = false;
            StepAction::Halve
        } else if prop_step < 0.5 * step && !self.doubled_step {
            self.doubled_step = true;
            StepAction::Double
        } else {
            self.doubled_step = false;
            StepAction::Keep
        }
    }
}

impl<P> MultiStep<MultiNumerovDataDMatrix<'_, P>> for MultiRatioNumerovStep<DMatrix<f64>>
where 
    P: Potential<Space = DMatrix<f64>>
{
    fn step(&mut self, data: &mut MultiNumerovDataDMatrix<P>) {
        data.r += data.dr;

        self.buffer1.zip_zip_apply(&data.unit, &data.current_g_func, |b1, u, c| {
            *b1 = u + data.dr * data.dr / 12. * c
        });

        self.f3.zip_apply(&self.buffer1, |f3, b1| *f3 = b1);
        self.f3.try_inverse_mut();

        data.psi2.zip_apply(&data.psi1, |p2, p1| *p2 = p1);
        data.psi2.try_inverse_mut();

        self.f2.mul_to(&data.psi2, &mut self.buffer2);
        self.buffer2.zip_zip_apply(&data.unit, &self.f1, |b2, u, f1| {
            *b2 = 12. * u - 10. * f1 - *b2
        });
        self.f3.mul_to(&self.buffer2, &mut data.psi2);

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        swap(&mut self.f1, &mut self.buffer1);

        swap(&mut data.psi2, &mut data.psi1);
    }

    fn halve_step(&mut self, data: &mut MultiNumerovDataDMatrix<P>) {
        data.dr /= 2.0;

        self.f2.zip_apply(&data.unit, |f2, u| {
            *f2 = *f2 / 4. + 0.75 * u
        });

        self.f1.zip_apply(&data.unit, |f1, u| {
            *f1 = *f1 / 4. + 0.75 * u
        });

        data.get_g_func(data.r - data.dr, &mut self.buffer1);
        self.buffer1.zip_apply(&data.unit, |b1, u| {
            *b1 = 2. * u - data.dr * data.dr * 10. / 12. * *b1
        });

        self.buffer2.zip_apply(&self.buffer1, |b2, b1| *b2 = b1);
        self.buffer2.try_inverse_mut();

        self.f2.zip_zip_apply(&data.unit, &self.buffer1, |f2, u, b1| {
            *f2 = 1.2 * u - b1 / 10.
        });

        self.f1.mul_to(&data.psi1, &mut self.buffer1);
        self.buffer1 += &self.f2;
        self.buffer2.mul_to(&self.buffer1, &mut data.psi2);

        self.buffer1.zip_apply(&data.psi2, |b1, p2| *b1 = p2);
        self.buffer1.try_inverse_mut();

        data.psi1.mul_to(&self.buffer1, &mut self.buffer2);
        swap(&mut data.psi1, &mut self.buffer2);
    }

    fn double_step(&mut self, data: &mut MultiNumerovDataDMatrix<P>) {
        data.dr *= 2.;

        self.f2.zip_zip_apply(&data.unit, &self.f3, |f2, u, f3| {
            *f2 = 4. * f3 - 3. * u
        });

        self.f1.zip_apply(&data.unit, |f1, u| {
            *f1 = 4. * *f1 - 3. * u
        });

        data.psi1.mul_to(&data.psi2, &mut self.buffer1);
        swap(&mut self.buffer1, &mut data.psi1);
    }
}


impl<P> PropagatorModifier<MultiNumerovDataDMatrix<'_, P>> for WaveStorage<DMatrix<f64>> 
where 
    P: Potential<Space = DMatrix<f64>>
{
    fn before(&mut self, data: &mut MultiNumerovDataDMatrix<'_, P>, r_stop: f64) {
        match &mut self.sampling {
            SampleConfig::Step(value) => {
                *value = (data.r - r_stop).abs() / self.capacity as f64
            },
            _ => {},
        }

        self.rs.push(data.r);
    }

    fn after_step(&mut self, data: &mut MultiNumerovDataDMatrix<'_, P>) {
        self.last_value = &data.psi1 * &self.last_value;

        match &mut self.sampling {
            SampleConfig::Each(sample_each) => {
                self.counter += 1;
                if self.counter % *sample_each == 0 {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value.clone());
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
                        .map(|(_, w)| w.clone())
                        .collect();
                }

            },
            SampleConfig::Step(sample_step) => {
                if (data.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value.clone());
                }
            },
        }
    }
}