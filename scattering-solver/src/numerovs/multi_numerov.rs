use crate::{
    boundary::{Asymptotic, Boundary},
    numerovs::{
        numerov_modifier::{PropagatorModifier, SampleConfig, WaveStorage},
        propagator::{
            MultiStep, MultiStepRule, Numerov, NumerovResult, PropagatorData, StepAction, StepRule,
        },
    },
    observables::s_matrix::SMatrix,
    potentials::{
        dispersion_potential::Dispersion,
        potential::{MatPotential, SimplePotential},
    },
    utility::{inverse_inplace, inverse_inplace_det},
};
use faer::{
    Mat, MatMut,
    linalg::matmul::matmul,
    prelude::{SolverCore, c64},
    unzipped, zipped,
};
use quantum::{
    params::particles::Particles,
    units::{Au, energy_units::Energy, mass_units::Mass},
    utility::{ratio_riccati_i, ratio_riccati_k, riccati_j, riccati_n},
};

use core::f64;
use std::{f64::consts::PI, mem::swap};

use super::{dummy_numerov::DummyMultiStep, numerov_modifier::ScatteringVsDistance};

pub type MultiRatioNumerov<'a, P, S> = Numerov<MultiNumerovData<'a, P>, S, MultiRatioNumerovStep>;

pub struct MultiRatioNumerovStep {
    f1: Mat<f64>,
    f2: Mat<f64>,
    f3: Mat<f64>,

    buffer1: Mat<f64>,
    buffer2: Mat<f64>,
}

pub struct MultiNumerovData<'a, P>
where
    P: MatPotential,
{
    pub r: f64,
    pub dr: f64,

    pub potential: &'a P,
    pub(super) mass: f64,
    pub(super) energy: f64,

    // todo! we assume diagonality of the asymptotic potential
    pub(super) asymptotic: &'a Asymptotic,
    pub(super) centrifugal_prop: Dispersion,

    pub(super) potential_buffer: Mat<f64>,
    pub(super) unit: Mat<f64>,
    pub(super) current_g_func: Mat<f64>,
    pub(super) psi2_det: f64,

    pub psi1: Mat<f64>,
    pub(super) psi2: Mat<f64>,

    pub(super) perm_buffer: Vec<usize>,
    pub(super) perm_inv_buffer: Vec<usize>,
}

impl<P> Clone for MultiNumerovData<'_, P>
where
    P: MatPotential,
{
    fn clone(&self) -> Self {
        Self { 
            r: self.r, 
            dr: self.dr, 
            potential: self.potential, 
            mass: self.mass, 
            energy: self.energy, 
            asymptotic: self.asymptotic, 
            centrifugal_prop: self.centrifugal_prop.clone(), 
            potential_buffer: self.potential_buffer.clone(), 
            unit: self.unit.clone(), 
            current_g_func: self.current_g_func.clone(), 
            psi2_det: self.psi2_det, 
            psi1: self.psi1.clone(), 
            psi2: self.psi2.clone(), 
            perm_buffer: self.perm_buffer.clone(), 
            perm_inv_buffer: self.perm_inv_buffer.clone() 
        }
    }
}

impl<P> MultiNumerovData<'_, P>
where
    P: MatPotential,
{
    pub fn get_g_func(&mut self, r: f64, out: MatMut<f64>) {
        self.potential.value_inplace(r, &mut self.potential_buffer);

        let centr_prop = self.centrifugal_prop.value(r);
        self.potential_buffer
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.asymptotic.centrifugal.iter())
            .for_each(|(x, l)| *x += (l.0 * (l.0 + 1)) as f64 * centr_prop);

        zipped!(out, self.unit.as_ref(), self.potential_buffer.as_ref())
            .for_each(|unzipped!(o, u, p)| *o = 2.0 * self.mass * (self.energy * u - p));
    }

    pub fn calculate_s_matrix(&self) -> SMatrix {
        let size = self.potential.size();
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;
        let wave_ratio = self.psi1.as_ref();

        // todo! check if it is better to include centrifugal barrier or not
        let asymptotic = &self.asymptotic.channel_energies;

        let is_open_channel = asymptotic
            .iter()
            .map(|&val| val < self.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .iter()
            .map(|&val| (2.0 * self.mass * (self.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = Mat::zeros(size, size);
        let mut j_prev_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_prev_last = Mat::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            let l = self.asymptotic.centrifugal[i].0;
            if is_open_channel[i] {
                j_last[(i, i)] = riccati_j(l, momentum * r_last) / momentum.sqrt();
                j_prev_last[(i, i)] = riccati_j(l, momentum * r_prev_last) / momentum.sqrt();
                n_last[(i, i)] = riccati_n(l, momentum * r_last) / momentum.sqrt();
                n_prev_last[(i, i)] = riccati_n(l, momentum * r_prev_last) / momentum.sqrt();
            } else {
                j_last[(i, i)] = ratio_riccati_i(l, momentum * r_last, momentum * r_prev_last);
                j_prev_last[(i, i)] = 1.0;
                n_last[(i, i)] = ratio_riccati_k(l, momentum * r_last, momentum * r_prev_last);
                n_prev_last[(i, i)] = 1.0;
            }
        }

        let denominator = (wave_ratio * n_prev_last - n_last).partial_piv_lu();
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
        let entrance = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == self.asymptotic.entrance)
            .expect("Closed entrance channel")
            .0;

        SMatrix::new(s_matrix, momenta[self.asymptotic.entrance], entrance)
    }
}

impl<'a, P, S, M> Numerov<MultiNumerovData<'a, P>, S, M>
where
    P: MatPotential,
    S: StepRule<MultiNumerovData<'a, P>>,
    M: MultiStep<MultiNumerovData<'a, P>>,
{
    pub fn get_result(&self) -> NumerovResult<Mat<f64>> {
        NumerovResult {
            r_last: self.data.r,
            dr: self.data.dr,
            wave_ratio: self.data.psi1.clone(),
        }
    }
}

impl<'a, P, S> Numerov<MultiNumerovData<'a, P>, S, MultiRatioNumerovStep>
where
    P: MatPotential,
    S: StepRule<MultiNumerovData<'a, P>>,
{
    pub fn new(
        potential: &'a P,
        particles: &'a Particles,
        step_rules: S,
        boundary: Boundary<Mat<f64>>,
    ) -> Self {
        let mass = particles
            .get::<Mass<Au>>()
            .expect("no reduced mass parameter Mass<Au> found in particles")
            .to_au();
        let mut energy = particles
            .get::<Energy<Au>>()
            .expect("no collision energy Energy<Au> found in particles")
            .to_au();

        let asymptotic = particles
            .get::<Asymptotic>()
            .expect("no Asymptotic found in particles for multi channel numerov problem");
        let centrifugal_prop = Dispersion::new(1. / (2. * mass), -2);

        energy += asymptotic.channel_energies[asymptotic.entrance];

        let size = potential.size();

        let r = boundary.r_start;
        let mut data = MultiNumerovData {
            r,
            dr: 0.,
            potential,
            centrifugal_prop,
            asymptotic,
            mass,
            energy,
            potential_buffer: Mat::zeros(size, size),
            unit: Mat::identity(size, size),
            current_g_func: Mat::zeros(size, size),
            psi1: Mat::zeros(size, size),
            psi2: Mat::zeros(size, size),
            perm_buffer: vec![0; size],
            perm_inv_buffer: vec![0; size],
            psi2_det: 0.0,
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
            buffer2: Mat::zeros(size, size),
        };

        Self {
            data,
            step_rules,
            multi_step,
        }
    }
}

impl<'a, P, S> Numerov<MultiNumerovData<'a, P>, S, MultiRatioNumerovStep>
where
    P: MatPotential,
    S: StepRule<MultiNumerovData<'a, P>> + Clone,
{
    pub fn as_dummy(&self) -> Numerov<MultiNumerovData<'a, P>, S, DummyMultiStep<MultiNumerovData<'a, P>>> {
        Numerov {
            data: self.data.clone(),
            step_rules: self.step_rules.clone(),
            multi_step: DummyMultiStep::default(),
        }
    }
}

impl<P> PropagatorData for MultiNumerovData<'_, P>
where
    P: MatPotential,
{
    fn step_size(&self) -> f64 {
        self.dr
    }

    fn current_g_func(&mut self) {
        self.potential
            .value_inplace(self.r + self.dr, &mut self.potential_buffer);

        let centr_prop = self.centrifugal_prop.value(self.r + self.dr);
        self.potential_buffer
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.asymptotic.centrifugal.iter())
            .for_each(|(x, l)| *x += (l.0 * (l.0 + 1)) as f64 * centr_prop);

        zipped!(
            self.current_g_func.as_mut(),
            self.unit.as_ref(),
            self.potential_buffer.as_ref()
        )
        .for_each(|unzipped!(c, u, p)| *c = 2.0 * self.mass * (self.energy * u - p));
    }

    fn crossed_distance(&self, r: f64) -> bool {
        self.dr.signum() * (r - self.r) <= 0.0
    }
}

impl<P> StepRule<MultiNumerovData<'_, P>> for MultiStepRule<MultiNumerovData<'_, P>>
where
    P: MatPotential,
{
    fn get_step(&self, data: &MultiNumerovData<P>) -> f64 {
        let max_g_val = data
            .current_g_func
            .diagonal()
            .column_vector()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let lambda = 2. * PI / max_g_val.abs().sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }

    fn assign(&mut self, data: &MultiNumerovData<P>) -> StepAction {
        let prop_step = data.step_size().abs();
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

impl<P> MultiStep<MultiNumerovData<'_, P>> for MultiRatioNumerovStep
where
    P: MatPotential,
{
    fn step(&mut self, data: &mut MultiNumerovData<P>) {
        data.r += data.dr;

        zipped!(
            self.buffer1.as_mut(),
            data.unit.as_ref(),
            data.current_g_func.as_ref()
        )
        .for_each(|unzipped!(b1, u, c)| *b1 = u + data.dr * data.dr / 12. * c);

        inverse_inplace(
            self.buffer1.as_ref(),
            self.f3.as_mut(),
            &mut data.perm_buffer,
            &mut data.perm_inv_buffer,
        );

        data.psi2_det = inverse_inplace_det(
            data.psi1.as_ref(),
            data.psi2.as_mut(),
            &mut data.perm_buffer,
            &mut data.perm_inv_buffer,
        );
        matmul(
            self.buffer2.as_mut(),
            self.f2.as_ref(),
            data.psi2.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );
        zipped!(self.buffer2.as_mut(), data.unit.as_ref(), self.f1.as_ref())
            .for_each(|unzipped!(b2, u, f1)| *b2 = 12. * u - 10. * f1 - *b2);
        matmul(
            data.psi2.as_mut(),
            self.f3.as_ref(),
            self.buffer2.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        swap(&mut self.f1, &mut self.buffer1);

        swap(&mut data.psi2, &mut data.psi1);
    }

    fn halve_step(&mut self, data: &mut MultiNumerovData<P>) {
        data.dr /= 2.0;

        zipped!(self.f2.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(f2, u)| *f2 = *f2 / 4. + 0.75 * u);

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(f1, u)| *f1 = *f1 / 4. + 0.75 * u);

        data.get_g_func(data.r - data.dr, self.buffer1.as_mut());
        zipped!(self.buffer1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(b1, u)| *b1 = 2. * u - data.dr * data.dr * 10. / 12. * *b1);

        inverse_inplace(
            self.buffer1.as_ref(),
            self.buffer2.as_mut(),
            &mut data.perm_buffer,
            &mut data.perm_inv_buffer,
        );

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.buffer1.as_ref())
            .for_each(|unzipped!(f2, u, b1)| *f2 = 1.2 * u - b1 / 10.);

        matmul(
            self.buffer1.as_mut(),
            self.f1.as_ref(),
            data.psi1.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );
        self.buffer1 += self.f2.as_ref();
        matmul(
            data.psi2.as_mut(),
            self.buffer2.as_ref(),
            self.buffer1.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );

        inverse_inplace(
            data.psi2.as_ref(),
            self.buffer1.as_mut(),
            &mut data.perm_buffer,
            &mut data.perm_inv_buffer,
        );

        matmul(
            self.buffer2.as_mut(),
            data.psi1.as_ref(),
            self.buffer1.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );
        swap(&mut data.psi1, &mut self.buffer2);
    }

    fn double_step(&mut self, data: &mut MultiNumerovData<P>) {
        data.dr *= 2.;

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.f3.as_ref())
            .for_each(|unzipped!(f2, u, f3)| *f2 = 4.0 * f3 - 3. * u);

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(f1, u)| *f1 = 4.0 * *f1 - 3. * u);

        matmul(
            self.buffer1.as_mut(),
            data.psi1.as_ref(),
            data.psi2.as_ref(),
            None,
            1.,
            faer::Parallelism::None,
        );
        swap(&mut self.buffer1, &mut data.psi1);
    }
}

impl<P> PropagatorModifier<MultiNumerovData<'_, P>> for WaveStorage<Mat<f64>>
where
    P: MatPotential,
{
    fn before(&mut self, data: &mut MultiNumerovData<'_, P>, r_stop: f64) {
        if let SampleConfig::Step(value) = &mut self.sampling {
            *value = (data.r - r_stop).abs() / self.capacity as f64
        }

        self.rs.push(data.r);
    }

    fn after_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
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
                if (data.r - self.rs.last().unwrap()).abs() > *sample_step {
                    self.rs.push(data.r);
                    self.waves.push(self.last_value.clone());
                }
            }
        }
    }
}

impl<P> PropagatorModifier<MultiNumerovData<'_, P>> for ScatteringVsDistance<SMatrix>
where
    P: MatPotential,
{
    fn before(&mut self, _data: &mut MultiNumerovData<'_, P>, r_stop: f64) {
        self.take_per = (r_stop - self.r_min).abs() / (self.capacity as f64);
    }

    fn after_step(&mut self, data: &mut MultiNumerovData<'_, P>) {
        if data.r < self.r_min {
            return;
        }

        let append = self
            .distances
            .last()
            .is_none_or(|r| (r - data.r).abs() >= self.take_per);

        if append {
            let s = data.calculate_s_matrix();
            self.distances.push(data.r);
            self.s_matrices.push(s)
        }
    }
}
