use scattering_solver::potentials::potential::{Potential, SimplePotential, SubPotential};
use spline_interpolation::{SplineBuilder, UniformSpline};


#[derive(Debug)]
pub struct PotentialArray {
    pub distances: Vec<f64>,
    pub potentials: Vec<(u32, Vec<f64>)>,
}

impl PotentialArray {
    pub fn new(distances: Vec<f64>, potentials: Vec<(u32, Vec<f64>)>) -> Self {
        Self {
            distances,
            potentials,
        }
    }
}

pub fn interpolate_potentials(pot_array: &PotentialArray, degree: u32) -> Vec<(u32, InterpolatedPotential)> {
    let mut interpolated = Vec::new();

    for (lambda, potential) in &pot_array.potentials {
        let spline = SplineBuilder::new(&pot_array.distances, &potential)
            .with_degree(degree)
            .build();

        let interp = InterpolatedPotential(spline);

        interpolated.push((*lambda, interp))
    }

    interpolated
}

#[derive(Clone)]
pub struct InterpolatedPotential(pub UniformSpline);

impl Potential for InterpolatedPotential {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        *value = self.0.eval(r);
    }

    fn size(&self) -> usize {
        1
    }
}

impl SubPotential for InterpolatedPotential {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        *value += self.0.eval(r)
    }
}

pub struct TransitionedPotential<P, V, F>
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    near: P,
    far: V,
    transition: F
}

impl<P, V, F> TransitionedPotential<P, V, F> 
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    pub fn new(near: P, far: V, transition: F) -> Self {
        Self {
            near,
            far,
            transition,
        }
    }
}

impl<P, V, F> Potential for TransitionedPotential<P, V, F> 
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        let a = (self.transition)(r);
        assert!(a >= 0. && a <= 1.);

        if a == 0. {
            self.far.value_inplace(r, value);
        } else if a == 1. {
            self.near.value_inplace(r, value);
        } else {
            *value = a * self.near.value(r) + (1. - a) * self.far.value(r)
        }
    }

    fn size(&self) -> usize {
        1
    }
}

impl<P, V, F> SubPotential for TransitionedPotential<P, V, F> 
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        let a = (self.transition)(r);
        assert!(a >= 0. && a <= 1.);

        if a == 0. {
            *value += self.far.value(r);
        } else if a == 1. {
            *value += self.near.value(r);
        } else {
            *value += a * self.near.value(r) + (1. - a) * self.far.value(r)
        }
    }
}