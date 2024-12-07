use std::f64::consts::PI;

use num::complex::Complex64;
use quantum::{params::particles::Particles, units::{energy_units::Energy, mass_units::Mass, Au}, utility::{asymptotic_bessel_j, asymptotic_bessel_n}};

use crate::{boundary::Boundary, observables::s_matrix::SingleSMatrix, potentials::{dispersion_potential::Dispersion, potential::SimplePotential, potential_factory::create_centrifugal}, utility::AngularSpin};

use super::propagator::{MultiStep, MultiStepRule, Numerov, NumerovResult, PropagatorData, StepAction, StepRule};

pub type SingleRatioNumerov<'a, P> = Numerov<
    SingleNumerovData<'a, P>, 
    MultiStepRule<SingleNumerovData<'a, P>>, 
    SingleRatioNumerovStep
>;

impl<'a, P, S, M> Numerov<SingleNumerovData<'a, P>, S, M> 
where 
    P: SimplePotential,
    S: StepRule<SingleNumerovData<'a, P>>,
    M: MultiStep<SingleNumerovData<'a, P>>
{
    pub fn get_result(&self) -> NumerovResult<f64> {
        NumerovResult {
            r_last: self.data.r,
            dr: self.data.dr,
            wave_ratio: self.data.psi1,
        }
    }
}

impl<'a, P, S> Numerov<SingleNumerovData<'a, P>, S, SingleRatioNumerovStep> 
where 
    P: SimplePotential,
    S: StepRule<SingleNumerovData<'a, P>>,
{
    pub fn new(potential: &'a P, particles: &Particles, step_rules: S, boundary: Boundary<f64>) -> Self {
        let mut data = SingleNumerovData::new(potential, particles);
        let r = boundary.r_start;
        data.r = r;
        data.current_g_func();
        
        let dr = match boundary.direction {
            crate::boundary::Direction::Inwards => -step_rules.get_step(&data),
            crate::boundary::Direction::Outwards => step_rules.get_step(&data),
            crate::boundary::Direction::Step(dr) => dr,
        };

        data.dr = dr;
    
        data.psi1 = boundary.start_value;
        data.psi2 = boundary.before_value;
    
        let f3 = 1. + dr * dr / 12. * data.get_g_func(r - 2. * dr);
        let f2 = 1. + dr * dr / 12. * data.get_g_func(r - dr);
        let f1 = 1. + dr * dr / 12. * data.current_g_func;

        let multi_step = SingleRatioNumerovStep {
            f1,
            f2,
            f3,
        };

        Self {
            data,
            step_rules,
            multi_step,
        }
    }
}

#[derive(Clone)]
pub struct SingleNumerovData<'a, P>
where 
    P: SimplePotential
{
    pub(super) r: f64,
    pub(super) dr: f64,

    pub(super) potential: &'a P,
    pub(super) mass: f64,
    pub(super) energy: f64,
    pub(super) l: AngularSpin,
    centrifugal: Option<Dispersion>,

    current_g_func: f64,

    pub(super) psi1: f64,
    psi2: f64,
}

impl<'a, P> SingleNumerovData<'a, P> 
where 
    P: SimplePotential
{
    pub fn new(potential: &'a P, particles: &Particles) -> Self {
        let mass = particles.get::<Mass<Au>>()
            .expect("no reduced mass parameter Mass<Au> found in particles")
            .to_au();
        let energy = particles.get::<Energy<Au>>()
            .expect("no collision energy Energy<Au> found in particles")
            .to_au();

        let l = particles.get::<AngularSpin>().unwrap_or(&AngularSpin(0));
        let centrifugal = create_centrifugal(mass, *l);

        Self {
            r: 0.,
            dr: 0.,
            potential,
            centrifugal,
            mass,
            energy,
            l: *l,
            current_g_func: 0.,
            psi1: 0.,
            psi2: 0.,
        }
    }

    pub fn calculate_s_matrix(&self) -> Result<SingleSMatrix, String> {
        let r_last = self.r;
        let r_prev_last = self.r - self.dr;
        let wave_ratio = self.psi1;

        let asymptotic = self.potential_value(r_last);

        let momentum = (2.0 * self.mass * (self.energy - asymptotic)).sqrt();
        if momentum.is_nan() {
            return Err("closed channel".to_string())
        }

        let j_last = asymptotic_bessel_j(momentum * r_last, self.l.0);
        let j_prev_last = asymptotic_bessel_j(momentum * r_prev_last, self.l.0);
        let n_last = asymptotic_bessel_n(momentum * r_last, self.l.0);
        let n_prev_last = asymptotic_bessel_n(momentum * r_prev_last, self.l.0);

        let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);

        let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        Ok(SingleSMatrix::new(s_matrix, momentum))
    }

    pub fn potential_value(&self, r: f64) -> f64 {
        let mut value = self.potential.value(r);

        if let Some(centr) = &self.centrifugal {
            value += centr.value(r);
        }

        value
    }

    fn get_g_func(&mut self, r: f64) -> f64 {
        2.0 * self.mass * (self.energy - self.potential_value(r))
    }
}

impl<P> PropagatorData for SingleNumerovData<'_, P> 
where 
    P: SimplePotential
{
    fn step_size(&self) -> f64 {
        self.dr
    }
    
    fn current_g_func(&mut self) {
        self.current_g_func = 2.0 * self.mass * (self.energy - self.potential_value(self.r + self.dr));
    }

    fn advance(&mut self) {
        self.r += self.dr;
    }
    
    fn crossed_distance(&self, r: f64) -> bool {
        self.dr.signum() * (r - self.r) <= 0.0
    }
}

impl<P> StepRule<SingleNumerovData<'_, P>> for MultiStepRule<SingleNumerovData<'_, P>>
where 
    P: SimplePotential
{
    fn get_step(&self, data: &SingleNumerovData<P>) -> f64 {
        let lambda = 2. * PI / data.current_g_func.abs().sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }
    
    fn assign(&mut self, data: &SingleNumerovData<P>) -> StepAction {
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

#[derive(Default)]
pub struct SingleRatioNumerovStep
{
    f1: f64,
    f2: f64,
    f3: f64,
}

impl<P> MultiStep<SingleNumerovData<'_, P>> for SingleRatioNumerovStep 
where 
    P: SimplePotential
{
    fn step(&mut self, data: &mut SingleNumerovData<P>) {
        data.r += data.dr;

        let f = 1.0 + data.dr * data.dr * data.current_g_func / 12.0;
        let psi = (12.0 - 10.0 * self.f1 - self.f2 / data.psi1) / f;

        self.f3 = self.f2;
        self.f2 = self.f1;
        self.f1 = f;

        data.psi2 = data.psi1;
        data.psi1 = psi;
    }

    fn halve_step(&mut self, data: &mut SingleNumerovData<P>) {
        data.dr /= 2.;

        let f = 1.0 + data.dr * data.dr * data.get_g_func(data.r - data.dr) / 12.0;
        self.f1 = self.f1 / 4.0 + 0.75;
        self.f2 = self.f2 / 4.0 + 0.75;

        let psi = (self.f1 * data.psi1 + self.f2) / (12.0 - 10.0 * f);
        self.f2 = f;

        data.psi2 = psi;
        data.psi1 /= psi;
    }

    fn double_step(&mut self, data: &mut SingleNumerovData<P>) {
        data.dr *= 2.0;

        self.f2 = 4.0 * self.f3 - 3.0;
        self.f1 = 4.0 * self.f1 - 3.0;

        data.psi1 *= data.psi2;
    }
}