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

/// Returns the legendre polynomials up to order `j` at `x`.
pub fn legendre_polynomials(j: u32, x: f64) -> Vec<f64> {
    let j = j as usize;
    let mut p = vec![0.0; j + 1];

    p[0] = 1.0;
    p[1] = x;

    for i in 2..=j {
        p[i] = ((2 * i - 1) as f64 / i as f64) * x * p[i - 1] - ((i - 1) as f64 / i as f64) * p[i - 2];
    }

    p
}

/// Returns the associated legendre polynomials up to `j` at `x`.
pub fn associated_legendre_polynomials(j: u32, m: i32, x: f64) -> Vec<f64> {
    let m_u = m.unsigned_abs();
    if m == 0 {
        return legendre_polynomials(j, x)
    }

    let j = j as usize;
    let m_u = m_u as usize;

    let mut p = vec![0.0; j + 1];

    p[m_u] = (-1.0f64).powi(m) * double_factorial(2 * m_u as u32 - 1) * (1. - x * x).powf(m_u as f64 / 2.0);

    if m < 0 {
        p[m_u] *= negate_m(m_u as u32, m_u as i32);
        p[m_u+1] = x * p[m_u]
    } else {
        p[m_u+1] = x * (2. * m_u as f64 + 1.) * p[m_u]
    }

    for i in m_u+2..=j {
        let l = i - 1;
        
        p[i] = (((2 * l + 1) as f64) * x * p[i - 1] - (l as f64 + m as f64) * p[i - 2]) / (l as f64 - m as f64 + 1.);
    }

    p
}

pub fn double_factorial(n: u32) -> f64 {
    assert!(n < 100);
    if n == 0 {
        return 1.;
    }

    let mut value = n as f64;
    for k in (2..n-1).rev().step_by(2) {
        value *= k as f64
    }

    value
}

fn negate_m(l: u32, m: i32) -> f64 {
    assert!(m < 50);

    if m > 0 {
        let mut value = (-1.0f64).powi(m);
        let min = l as i32 - m;
        let max = l as i32 + m;

        for k in min..max {
            value /= k as f64 + 1.
        }

        value
    } else if m < 0 {
        let mut value = (-1.0f64).powi(m);
        let min = l as i32 + m;
        let max = l as i32 - m;

        for k in min..max {
            value *= k as f64 + 1.
        }

        value
    } else {
        1.0
    }
}

pub fn normalization(l: u32, m: i32) -> f64 {
    assert!(m < 50);

    let norm2 = if m > 0 {
        let mut value = 1.0;
        let min = l as i32 - m;
        let max = l as i32 + m;

        for k in min..max {
            value /= k as f64 + 1.;
            if value.is_nan() || value < 0.{
                println!("{} {} {}", l, m, k)
            }
        }

        value
    } else if m < 0 {
        let mut value = 1.0;
        let min = l as i32 + m;
        let max = l as i32 - m;

        for k in min..max {
            value *= k as f64 + 1.
        }

        value
    } else {
        1.0
    };

    (norm2 * (l as f64 + 0.5)).sqrt()
}