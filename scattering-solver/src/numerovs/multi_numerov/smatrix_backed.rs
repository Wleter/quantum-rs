use std::{f64::consts::PI, mem::swap};

use faer::{linalg::matmul::matmul, prelude::c64, solvers::SolverCore, unzipped, zipped, Mat, MatMut};
use quantum::{params::particles::Particles, units::{energy_units::Energy, mass_units::Mass, Au}, utility::{asymptotic_bessel_j, asymptotic_bessel_n, bessel_j_ratio, bessel_n_ratio}};
use crate::{boundary::Boundary, numerovs::{numerov_modifier::{PropagatorModifier, SampleConfig, WaveStorage}, propagator::{MultiStep, MultiStepRule, Numerov, NumerovResult, PropagatorData, StepAction, StepRule}}, observables::s_matrix::faer::FaerSMatrix, potentials::{dispersion_potential::Dispersion, potential::{Potential, SimplePotential}, potential_factory::create_centrifugal}, utility::{faer::inverse_inplace, AngularSpin}};
use super::MultiRatioNumerovStep;

#[derive(Clone)]
pub struct MultiNumerovDataSMatrix<'a, P, const N: usize>
where 
    P: Potential<Space = Mat<f64>>
{
    pub(super) r: f64,
    pub(super) dr: f64,

    pub(super) potential: &'a P,
    pub(super) centrifugal: Option<Dispersion>,
    pub(super) mass: f64,
    pub(super) energy: f64,
    pub(super) l: AngularSpin,

    pub(super) potential_buffer: Mat<f64>,
    pub(super) unit: Mat<f64>,
    pub(super) current_g_func: Mat<f64>,

    pub(super) psi1: Mat<f64>,
    pub(super) psi2: Mat<f64>,

    pub(super) perm: Vec<usize>,
    pub(super) perm_inv: Vec<usize>,
}


pub type DMatrixRatioNumerov<'a, P> = Numerov<
    MultiNumerovDataFaer<'a, P>, 
    MultiStepRule<MultiNumerovDataFaer<'a, P>>, 
    MultiRatioNumerovStep<Mat<f64>>
>;

impl<P> MultiNumerovDataFaer<'_, P> 
where 
    P: Potential<Space = Mat<f64>>
{
    pub fn get_g_func(&mut self, r: f64, out: MatMut<f64>) {
        self.potential.value_inplace(r, &mut self.potential_buffer);


        if let Some(centr) = &self.centrifugal {
            let centr = centr.value(r);

            self.potential_buffer.diagonal_mut()
                .column_vector_mut()
                .iter_mut()
                .for_each(|x| *x += centr)
        }

        zipped!(out, self.unit.as_ref(), self.potential_buffer.as_ref())
            .for_each(|unzipped!(mut o, u, p)| o.write(2.0 * self.mass * (self.energy * u.read() - p.read())));
    }

    pub fn calculate_s_matrix(&self, entrance: usize) -> FaerSMatrix {
        let size = self.potential.size();
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;
        let wave_ratio = self.psi1.as_ref();

        let mut asymptotic = Mat::zeros(size, size);
        self.potential.value_inplace(r_last, &mut asymptotic);

        // todo! assume diagonality of the asymptotic potential

        let is_open_channel = asymptotic
            .diagonal()
            .column_vector()
            .iter()
            .map(|&val| val < self.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .diagonal()
            .column_vector()
            .iter()
            .map(|&val| (2.0 * self.mass * (self.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = Mat::zeros(size, size);
        let mut j_prev_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_prev_last = Mat::zeros(size, size);

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

        let denominator = (&wave_ratio * n_prev_last - n_last).partial_piv_lu();
        let denominator = denominator.inverse();

        let k_matrix = -denominator * (wave_ratio * j_prev_last - j_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);

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

                red_ik_matrix[(i, j)] = c64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = Mat::<c64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).partial_piv_lu();
        let denominator = denominator.inverse();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel.iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == entrance)
            .expect("Closed entrance channel")
            .0;

        FaerSMatrix::new(s_matrix, momenta, entrance) // todo!
    }
}

impl<'a, P, S, M> Numerov<MultiNumerovDataFaer<'a, P>, S, M> 
where 
    P: Potential<Space = Mat<f64>>,
    S: StepRule<MultiNumerovDataFaer<'a, P>>,
    M: MultiStep<MultiNumerovDataFaer<'a, P>>
{
    pub fn get_result(&self) -> NumerovResult<Mat<f64>> {
        NumerovResult {
            r_last: self.data.r,
            dr: self.data.dr,
            wave_ratio: self.data.psi1.clone(),
        }
    }
}

impl<'a, P, S> Numerov<MultiNumerovDataFaer<'a, P>, S, MultiRatioNumerovStep<Mat<f64>>>
where 
    P: Potential<Space = Mat<f64>>,
    S: StepRule<MultiNumerovDataFaer<'a, P>>,
{
    pub fn new(potential: &'a P, particles: &Particles, step_rules: S, boundary: Boundary<Mat<f64>>) -> Self {
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
        let mut data = MultiNumerovDataFaer {
            r,
            dr: 0.,
            potential,
            centrifugal,
            l,
            mass,
            energy,
            potential_buffer: Mat::zeros(size, size),
            unit: Mat::identity(size, size),
            current_g_func: Mat::zeros(size, size),
            psi1: Mat::zeros(size, size),
            psi2: Mat::zeros(size, size),
            perm: vec![0; size],
            perm_inv: vec![0; size]
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
    
        let mut f3 = Mat::zeros(size, size);
        data.get_g_func(r - 2. * dr, f3.as_mut());

        let mut f2 = Mat::zeros(size, size);
        data.get_g_func(r - dr, f2.as_mut());

        let f3 = data.unit.as_ref() + dr * dr / 12. * f3;
        let f2 = data.unit.as_ref() + dr * dr / 12. * f2;
        let f1 = data.unit.as_ref() + dr * dr / 12. * &data.current_g_func;

        let multi_step = MultiRatioNumerovStep { 
            f1, 
            f2, 
            f3, 
            buffer1: Mat::zeros(size, size), 
            buffer2: Mat::zeros(size, size)
        };

        Self {
            data,
            step_rules,
            multi_step,
        }
    }
}

impl<P> PropagatorData for MultiNumerovDataFaer<'_, P> 
where 
    P: Potential<Space = Mat<f64>>
{
    fn step_size(&self) -> f64 {
        self.dr
    }
    
    fn current_g_func(&mut self) {
        self.potential.value_inplace(self.r + self.dr, &mut self.potential_buffer);

        if let Some(centr) = &self.centrifugal {
            let centr = centr.value(self.r + self.dr);

            self.potential_buffer.diagonal_mut()
                .column_vector_mut()
                .iter_mut()
                .for_each(|x| *x += centr)
        }

        zipped!(self.current_g_func.as_mut(), self.unit.as_ref(), self.potential_buffer.as_ref())
            .for_each(|unzipped!(mut c, u, p)| 
                c.write(2.0 * self.mass * (self.energy * u.read() - p.read()))
            );
    }

    fn advance(&mut self) {
        self.r += self.dr;
    }

    fn crossed_distance(&self, r: f64) -> bool {
        self.dr.signum() * (r - self.r) <= 0.0
    }
}

impl<P> StepRule<MultiNumerovDataFaer<'_, P>> for MultiStepRule<MultiNumerovDataFaer<'_, P>>
where 
    P: Potential<Space = Mat<f64>>
{
    fn get_step(&self, data: &MultiNumerovDataFaer<P>) -> f64 {
        let mut max_g_val = 0.;
        zipped!(data.current_g_func.as_ref())
            .for_each(|unzipped!(c)| max_g_val = f64::max(max_g_val, c.read().abs()));

        let lambda = 2. * PI / max_g_val.sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }
    
    fn assign(&mut self, data: &MultiNumerovDataFaer<P>) -> StepAction {
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

impl<P> MultiStep<MultiNumerovDataFaer<'_, P>> for MultiRatioNumerovStep<Mat<f64>>
where 
    P: Potential<Space = Mat<f64>>
{
    fn step(&mut self, data: &mut MultiNumerovDataFaer<P>) {
        data.r += data.dr;

        zipped!(self.buffer1.as_mut(), data.unit.as_ref(), data.current_g_func.as_ref())
            .for_each(|unzipped!(mut b1, u, c)| 
                b1.write(u.read() + data.dr * data.dr / 12. * c.read())
            );

        inverse_inplace(self.buffer1.as_ref(), self.f3.as_mut(), &mut data.perm, &mut data.perm_inv);

        inverse_inplace(data.psi1.as_ref(), data.psi2.as_mut(), &mut data.perm, &mut data.perm_inv);
        matmul(self.buffer2.as_mut(), self.f2.as_ref(), data.psi2.as_ref(), None, 1., faer::Parallelism::None);
        zipped!(self.buffer2.as_mut(), data.unit.as_ref(), self.f1.as_ref())
            .for_each(|unzipped!(mut b2, u, f1)| 
                b2.write(12. * u.read() - 10. * f1.read() - b2.read())
            );
        matmul(data.psi2.as_mut(), self.f3.as_ref(), self.buffer2.as_ref(), None, 1., faer::Parallelism::None);

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        swap(&mut self.f1, &mut self.buffer1);

        swap(&mut data.psi2, &mut data.psi1);
    }

    fn halve_step(&mut self, data: &mut MultiNumerovDataFaer<P>) {
        data.dr /= 2.0;

        zipped!(self.f2.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f2, u)| 
                f2.write(f2.read() / 4. + 0.75 * u.read())
            );

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f1, u)| 
                f1.write(f1.read() / 4. + 0.75 * u.read())
            );

        data.get_g_func(data.r - data.dr, self.buffer1.as_mut());
        zipped!(self.buffer1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut b1, u)| 
                b1.write(2. * u.read() - data.dr * data.dr * 10. / 12. * b1.read())
            );

        inverse_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut data.perm, &mut data.perm_inv);

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.buffer1.as_ref())
            .for_each(|unzipped!(mut f2, u, b1)| 
                f2.write(1.2 * u.read() - b1.read() / 10.)
            );

        matmul(self.buffer1.as_mut(), self.f1.as_ref(), data.psi1.as_ref(), None, 1., faer::Parallelism::None);
        self.buffer1 += self.f2.as_ref();
        matmul(data.psi2.as_mut(), self.buffer2.as_ref(), self.buffer1.as_ref(), None, 1., faer::Parallelism::None);

        inverse_inplace(data.psi2.as_ref(), self.buffer1.as_mut(), &mut data.perm, &mut data.perm_inv);

        matmul(self.buffer2.as_mut(), data.psi1.as_ref(), self.buffer1.as_ref(), None, 1., faer::Parallelism::None);
        swap(&mut data.psi1, &mut self.buffer2);
    }

    fn double_step(&mut self, data: &mut MultiNumerovDataFaer<P>) {
        data.dr *= 2.;

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.f3.as_ref())
            .for_each(|unzipped!(mut f2, u, f3)| 
                f2.write(4.0 * f3.read() - 3. * u.read())
            );

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f1, u)| 
                f1.write(4.0 * f1.read() - 3. * u.read())
            );

        matmul(self.buffer1.as_mut(), data.psi1.as_ref(), data.psi2.as_ref(), None, 1., faer::Parallelism::None);
        swap(&mut self.buffer1, &mut data.psi1);
    }
}


impl<P> PropagatorModifier<MultiNumerovDataFaer<'_, P>> for WaveStorage<Mat<f64>> 
where 
    P: Potential<Space = Mat<f64>>
{
    fn before(&mut self, data: &mut MultiNumerovDataFaer<'_, P>, r_stop: f64) {
        match &mut self.sampling {
            SampleConfig::Step(value) => {
                *value = (data.r - r_stop).abs() / self.capacity as f64
            },
            _ => {},
        }

        self.rs.push(data.r);
    }

    fn after_step(&mut self, data: &mut MultiNumerovDataFaer<'_, P>) {
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