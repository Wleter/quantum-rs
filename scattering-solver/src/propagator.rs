use faer::{Mat, unzipped, zipped};
use quantum::{
    params::particles::Particles,
    units::{Au, energy_units::Energy, mass_units::Mass},
};

use crate::{
    boundary::Asymptotic, numerovs::propagator_watcher::PropagatorWatcher,
    potentials::potential::Potential,
};

pub trait Repr<T> {}

#[derive(Clone, Debug, Default)]
pub struct Solution<R> {
    pub r: f64,
    pub dr: f64,
    pub sol: R,
    pub nodes: u64,
}

pub type SingleEquation<'a> = Equation<'a, f64>;
pub type CoupledEquation<'a> = Equation<'a, Mat<f64>>;

#[derive(Clone)]
pub struct Equation<'a, T> {
    pub potential: &'a dyn Potential<Space = T>,
    pub energy: f64,
    pub mass: f64,
    pub asymptotic: Asymptotic,

    buffered_w_matrix: T,
    pub(super) unit: T,
}

impl<'a> Equation<'a, Mat<f64>> {
    pub fn new<P: Potential<Space = Mat<f64>>>(
        potential: &'a P,
        mut energy: f64,
        mass: f64,
        asymptotic: Asymptotic,
    ) -> Self {
        energy += asymptotic.channel_energies[asymptotic.entrance];

        Self {
            potential,
            energy,
            mass,
            asymptotic,

            buffered_w_matrix: Mat::zeros(potential.size(), potential.size()),
            unit: Mat::identity(potential.size(), potential.size()),
        }
    }

    pub fn from_particles<P: Potential<Space = Mat<f64>>>(
        potential: &'a P,
        particles: &Particles,
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
            .expect("no Asymptotic found in particles for multi channel numerov problem")
            .clone();

        energy += asymptotic.channel_energies[asymptotic.entrance];

        Self {
            potential,
            energy,
            mass,
            asymptotic,

            buffered_w_matrix: Mat::zeros(potential.size(), potential.size()),
            unit: Mat::identity(potential.size(), potential.size()),
        }
    }

    pub fn w_matrix(&self, r: f64, out: &mut Mat<f64>) {
        self.potential.value_inplace(r, out);

        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.asymptotic.centrifugal.iter())
            .for_each(|(x, l)| *x += (l.0 * (l.0 + 1)) as f64 / (2. * self.mass * r * r));

        zipped!(out.as_mut(), self.unit.as_ref())
            .for_each(|unzipped!(o, u)| *o = 2.0 * self.mass * (self.energy * u - *o));
    }

    pub fn buffer_w_matrix(&mut self, r: f64) {
        self.potential.value_inplace(r, &mut self.buffered_w_matrix);

        self.buffered_w_matrix
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.asymptotic.centrifugal.iter())
            .for_each(|(x, l)| *x += (l.0 * (l.0 + 1)) as f64 / (2. * self.mass * r * r));

        zipped!(self.buffered_w_matrix.as_mut(), self.unit.as_ref())
            .for_each(|unzipped!(o, u)| *o = 2.0 * self.mass * (self.energy * u - *o));
    }

    // todo! check if including centrifugal term is good,
    // however then how to differentiate open channels with closed etc
    pub fn asymptotic(&self, _r: f64) -> &[f64] {
        &self.asymptotic.channel_energies
        // .iter()
        // .zip(&self.asymptotic.centrifugal)
        // .map(move |(e, AngMomentum(l))| {
        //     e + (l * (l + 1)) as f64 / (2. * self.mass * r * r)
        // })
        // .collect()
    }

    pub fn buffered_w_matrix(&self) -> &Mat<f64> {
        &self.buffered_w_matrix
    }
}

impl<'a> Equation<'a, f64> {
    pub fn new<P: Potential<Space = f64>>(
        potential: &'a P,
        mut energy: f64,
        mass: f64,
        asymptotic: Asymptotic,
    ) -> Self {
        energy += asymptotic.channel_energies[asymptotic.entrance];

        Self {
            potential,
            energy,
            mass,
            asymptotic,

            buffered_w_matrix: 0.,
            unit: 1.,
        }
    }

    pub fn from_particles<P: Potential<Space = f64>>(
        potential: &'a P,
        particles: &Particles,
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
            .unwrap_or(&Asymptotic::single_default())
            .clone();

        energy += asymptotic.channel_energies[asymptotic.entrance];

        Self {
            potential,
            energy,
            mass,
            asymptotic,

            buffered_w_matrix: 0.,
            unit: 1.,
        }
    }

    pub fn w_matrix(&self, r: f64) -> f64 {
        let mut value = 0.0;
        self.potential.value_inplace(r, &mut value);

        2.0 * self.mass * (self.energy - value)
    }

    pub fn buffer_w_matrix(&mut self, r: f64) {
        self.buffered_w_matrix = self.w_matrix(r)
    }

    pub fn buffered_w_matrix(&self) -> f64 {
        self.buffered_w_matrix
    }
}

pub trait Propagator<T, R: Repr<T>> {
    fn step(&mut self) -> &Solution<R>;

    fn propagate_to(&mut self, r: f64) -> &Solution<R>;

    fn propagate_to_with(
        &mut self,
        r: f64,
        modifier: &mut impl PropagatorWatcher<T, R>,
    ) -> &Solution<R>;
}

pub trait MultiStep<T, R: Repr<T>> {
    /// Performs a step with the same step size
    fn perform_step(&mut self, sol: &mut Solution<R>, eq: &Equation<T>);

    /// Halves the step size without actually performing a step
    fn halve_the_step(&mut self, sol: &mut Solution<R>, eq: &Equation<T>);

    /// Doubles the step size without actually performing a step
    fn double_the_step(&mut self, sol: &mut Solution<R>, eq: &Equation<T>);
}
