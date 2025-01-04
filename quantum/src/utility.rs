use std::{cmp::Ordering, f64::consts::FRAC_PI_2};

use crate::units::{energy_units::Energy, Au, Unit};

pub fn asymptotic_bessel_j(x: f64, l: u32) -> f64 {
    (x - FRAC_PI_2 * (l as f64)).sin()
}

pub fn asymptotic_bessel_n(x: f64, l: u32) -> f64 {
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

pub fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n as f64 - 1.0);

    for i in 0..n {
        result.push((10.0f64).powf(start + (i as f64) * step));
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

    match m.cmp(&0) {
        Ordering::Greater => {
            let mut value = (-1.0f64).powi(m);
            let min = l as i32 - m;
            let max = l as i32 + m;
    
            for k in min..max {
                value /= k as f64 + 1.
            }
    
            value
        },
        Ordering::Less => {
            let mut value = (-1.0f64).powi(m);
            let min = l as i32 + m;
            let max = l as i32 - m;
    
            for k in min..max {
                value *= k as f64 + 1.
            }
    
            value
        },
        Ordering::Equal => 1.0
    }
}

pub fn normalization(l: u32, m: i32) -> f64 {
    assert!(m < 50);

    let norm2 = match m.cmp(&0) {
        Ordering::Greater => {
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
        },
        Ordering::Less => {
            let mut value = 1.0;
            let min = l as i32 + m;
            let max = l as i32 - m;
    
            for k in min..max {
                value *= k as f64 + 1.
            }
    
            value
        },
        Ordering::Equal => 1.0
    };

    (norm2 * (l as f64 + 0.5)).sqrt()
}

/// Macro for crating cache with given name `$cache_name:ident` around the expression
/// It is used together with `cached_mel!` macro to cache subsequent calculations of the operator creation.
/// 
/// # Syntax
///
/// - `make_cache!($cache_name:ident, $body:expr)`
/// - `make_cache!($cache_name:ident => $capacity:expr, $body:expr)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$cache_name`: Name of the create cache.
/// - `$body`: body around which to create the cache.
///
/// ## Optional Arguments
/// - `$capacity` (optional): Initial capacity of the cache.
#[macro_export]
macro_rules! make_cache {
    ($cache_name:ident => $capacity:expr, $body:expr) => {{
        let mut $cache_name = std::collections::HashMap::with_capacity($capacity);
        $body
    }};
    ($cache_name:ident, $body:expr) => {{
        let mut $cache_name = std::collections::HashMap::new();
        $body
    }};
}

/// Macro for caching and retrieving matrix elements.
/// 
/// # Syntax
///
/// - `cached_mel!($cache_name:ident, $func:ident($($arg:expr),*))`
/// - `cached_mel!($cache_name:ident, |[$($arg:ident),*]| $body:block)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$cache_name`: Name of the used cache.
/// - `$func(arg)`: Function outputs to be cached with `arg` keys.
/// or - `|[$($arg:ident),*]| $body:block`: Closure to be cached 
///
/// # Matched Arms
///
/// ## Arm 1: `cached_mel!($cache_name, $func($($arg),*))`
/// Use this arm when you want to cache function
///
/// ## Arm 2: `cached_mel!($cache_name, |[$($arg),*]| $body)`
/// Use this arm when you want to cache a closure in an operator
#[macro_export]
macro_rules! cached_mel {
    ($cache_name:ident, $func:ident($($arg:expr),*)) => {{
        if let Some(result) = $cache_name.get(&($($arg),*)) {
            return *result;
        }

        let result = $func($($arg),*);
        $cache_name.insert(($($arg),*), result);
        result
    }};
    ($cache_name:ident, |[$($arg:ident),*]| $body:block) => {
        |[$($arg),*]| {
            if let Some(result) = $cache_name.get(&($($arg),*)) {
                return *result;
            }
    
            let result = $body;
            $cache_name.insert(($($arg),*), result);
            result
        }
    };
}

#[cfg(test)]
mod test {
    use std::{thread::sleep, time::{Duration, Instant}};

    fn long_computation<const N: usize>(a: [usize; N]) -> [usize; N] {
        sleep(Duration::from_millis(500));
        a
    }

    fn print_computations<const N: usize>(a: Vec<[usize; N]>, mut f: impl FnMut([usize; N]) -> [usize; N]) -> Vec<f64> {
        a.iter()
            .map(|&a| {
                let start = Instant::now();
                let a = f(a);
                let end = start.elapsed();
                println!("result {:?} time: {}", a, end.as_secs_f64());
                
                end.as_secs_f64()
            })
            .collect()
    }

    #[test]
    fn test_caching() {
        let values = vec![[3], [3], [4]];
        let durations = make_cache!(cache, print_computations(values, |a| {
            cached_mel!(cache, long_computation(a))
        }));
        assert!(durations[0] >= 0.5);
        assert!(durations[1] < 0.5);
        assert!(durations[2] >= 0.5);

        let values = vec![[3], [3], [4]];
        let durations = make_cache!(
            cache, 
            print_computations(values, cached_mel!(cache, |[a]| {
                long_computation([a])
            }))
        );
        assert!(durations[0] >= 0.5);
        assert!(durations[1] < 0.5);
        assert!(durations[2] >= 0.5);
    }
}