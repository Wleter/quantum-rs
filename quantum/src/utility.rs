use std::f64::consts::FRAC_PI_2;

use crate::units::{energy_units::Energy, Au, Unit};

pub fn asymptotic_bessel_j(x: f64, l: usize) -> f64 {
    (x - FRAC_PI_2 * (l as f64)).sin()
}

pub fn asymptotic_bessel_n(x: f64, l: usize) -> f64 {
    (x - FRAC_PI_2 * (l as f64)).cos()
}

pub fn bessel_j_ratio(x1: f64, x2: f64) -> f64 {
    (x1 - x2).exp() * (1.0 - (-2.0 * x1).exp()) / (1.0 - (-2.0 * x2).exp())
}

pub fn bessel_n_ratio(x1: f64, x2: f64) -> f64 {
    (x2 - x1).exp()
}

pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n as f64 - 1.0);

    for i in 0..n {
        result.push(start + (i as f64) * step);
    }

    result
}

pub fn unit_linspace<U: Unit>(start: Energy<U>, end: Energy<U>, n: usize) -> Vec<Energy<U>> {
    let start_au = start.to_au();
    let end_au = end.to_au();

    linspace(start_au, end_au, n)
        .into_iter()
        .map(|x| Energy(x, Au).to(start.unit()))
        .collect()
}
