use std::cmp::Ordering;

/// Creates evenly spaced grid of points [start, end] (including) with n points.
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

/// Creates logarithmically spaced grid of points [10^start, 10^end] (including) with n points.
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

/// Returns the legendre polynomials up to order `j` at `x`.
pub fn legendre_polynomials(j: u32, x: f64) -> Vec<f64> {
    let j = j as usize;
    let mut p = vec![0.0; j + 1];

    p[0] = 1.0;
    p[1] = x;

    for i in 2..=j {
        p[i] =
            ((2 * i - 1) as f64 / i as f64) * x * p[i - 1] - ((i - 1) as f64 / i as f64) * p[i - 2];
    }

    p
}

/// Returns the associated legendre polynomials up to order `j` at `x`.
pub fn associated_legendre_polynomials(j: u32, m: i32, x: f64) -> Vec<f64> {
    let m_u = m.unsigned_abs();
    if m == 0 {
        return legendre_polynomials(j, x);
    }

    let j = j as usize;
    let m_u = m_u as usize;

    let mut p = vec![0.0; j + 1];

    p[m_u] = (-1.0f64).powi(m)
        * double_factorial(2 * m_u as u32 - 1)
        * (1. - x * x).powf(m_u as f64 / 2.0);

    if m < 0 {
        p[m_u] *= negate_m(m_u as u32, m_u as i32);
        p[m_u + 1] = x * p[m_u]
    } else {
        p[m_u + 1] = x * (2. * m_u as f64 + 1.) * p[m_u]
    }

    for i in m_u + 2..=j {
        let l = i - 1;

        p[i] = (((2 * l + 1) as f64) * x * p[i - 1] - (l as f64 + m as f64) * p[i - 2])
            / (l as f64 - m as f64 + 1.);
    }

    p
}

pub fn double_factorial(n: u32) -> f64 {
    assert!(n < 100);
    if n == 0 {
        return 1.;
    }

    let mut value = n as f64;
    for k in (2..n - 1).rev().step_by(2) {
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
        }
        Ordering::Less => {
            let mut value = (-1.0f64).powi(m);
            let min = l as i32 + m;
            let max = l as i32 - m;

            for k in min..max {
                value *= k as f64 + 1.
            }

            value
        }
        Ordering::Equal => 1.0,
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
                if value.is_nan() || value < 0. {
                    println!("{} {} {}", l, m, k)
                }
            }

            value
        }
        Ordering::Less => {
            let mut value = 1.0;
            let min = l as i32 + m;
            let max = l as i32 - m;

            for k in min..max {
                value *= k as f64 + 1.
            }

            value
        }
        Ordering::Equal => 1.0,
    };

    (norm2 * (l as f64 + 0.5)).sqrt()
}

/// Calculates riccati bessel function of the first kind j_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j(n: u32, x: f64) -> f64 {
    bessel_recurrence(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x))
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n(n: u32, x: f64) -> f64 {
    bessel_recurrence(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x))
}

/// Calculates ratio of the riccati modified spherical bessel function of the first kind
/// (that is $sqrt(x) I_{n+1/2}(x)) at points `x_1`, `x_2`
///
/// "Handbook of Mathematical Functions" - eq. 10.2.2 (written as z * sqrt(pi/2z) I_{n+1/2}(z))
pub fn ratio_riccati_i(n: u32, x_1: f64, x_2: f64) -> f64 {
    let red_i_0 = |x| (1. - f64::exp(-2.0 * x)) / 2.0;
    let red_i_1 = |x| -(1. - f64::exp(-2.0 * x)) / (2.0 * x) + (1. + f64::exp(-2.0 * x)) / 2.0;

    // Calculates riccati I bessel without leading exponent
    let i_1 = modified_bessel_recurrence(n, x_1, red_i_0(x_1), red_i_1(x_1));
    let i_2 = modified_bessel_recurrence(n, x_2, red_i_0(x_2), red_i_1(x_2));

    f64::exp(x_1 - x_2) * i_1 / i_2
}

/// Calculates ratio of the riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x)) at points `x_1`, `x_2`
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn ratio_riccati_k(n: u32, x_1: f64, x_2: f64) -> f64 {
    let red_k_0 = |_| 1.0;
    let red_k_1 = |x| (1.0 + 1.0 / x);

    // Calculates riccati $(-1)^(n+1) * K$ bessel without leading exponent
    let k_1 = modified_bessel_recurrence(n, x_1, -red_k_0(x_1), red_k_1(x_1));
    let k_2 = modified_bessel_recurrence(n, x_2, -red_k_0(x_2), red_k_1(x_2));

    f64::exp(x_2 - x_1) * k_1 / k_2
}

/// Calculated f_{n+1}(x) given n, x, f_n(x), f_{n-1}(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19
fn bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
    if n == 0 {
        return f_0;
    }
    if n == 1 {
        return f_1;
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }

    f_k
}

/// Calculated f_{n+1}(x) given n, x, f_n(x), f_{n-1}(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn modified_bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
    if n == 0 {
        return f_0;
    }
    if n == 1 {
        return f_1;
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = f_k_1 - (2 * k + 1) as f64 / x * f_k;
        f_k_1 = f_k;
        f_k = f_new;
    }

    f_k
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
///     or - `|[$($arg:ident),*]| $body:block`: Closure to be cached
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

#[macro_export]
/// Asserts relative error |x - y| < x * err
///
/// # Syntax
///
/// - `assert_approx_eq!(x, y, err)`
macro_rules! assert_approx_eq {
    ($x:expr, $y:expr, $err:expr) => {
        if ($x - $y).abs() >= $x.abs() * $err {
            panic!("assertion failed\nleft side: {}\nright side: {}", $x, $y)
        }
    };
}

#[cfg(test)]
mod test {
    use std::{
        thread::sleep,
        time::{Duration, Instant},
    };

    use crate::utility::{ratio_riccati_i, ratio_riccati_k, riccati_n};

    use super::riccati_j;

    fn long_computation<const N: usize>(a: [usize; N]) -> [usize; N] {
        sleep(Duration::from_millis(500));
        a
    }

    fn print_computations<const N: usize>(
        a: Vec<[usize; N]>,
        mut f: impl FnMut([usize; N]) -> [usize; N],
    ) -> Vec<f64> {
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
        let durations = make_cache!(
            cache,
            print_computations(values, |a| { cached_mel!(cache, long_computation(a)) })
        );
        assert!(durations[0] >= 0.5);
        assert!(durations[1] < 0.5);
        assert!(durations[2] >= 0.5);

        let values = vec![[3], [3], [4]];
        let durations = make_cache!(
            cache,
            print_computations(values, cached_mel!(cache, |[a]| { long_computation([a]) }))
        );
        assert!(durations[0] >= 0.5);
        assert!(durations[1] < 0.5);
        assert!(durations[2] >= 0.5);
    }

    #[test]
    fn test_bessel() {
        assert_approx_eq!(riccati_j(5, 10.0), -0.555345, 1e-5);
        assert_approx_eq!(riccati_j(10, 10.0), 0.646052, 1e-5);

        assert_approx_eq!(riccati_n(5, 10.0), -0.938335, 1e-5);
        assert_approx_eq!(riccati_n(10, 10.0), 1.72454, 1e-5);

        assert_approx_eq!(ratio_riccati_i(5, 5.0, 10.0), 0.00157309, 1e-5);
        assert_approx_eq!(ratio_riccati_i(10, 5.0, 10.0), 0.00011066, 1e-5);

        assert_approx_eq!(ratio_riccati_k(5, 5.0, 10.0), 487.227, 1e-5);
        assert_approx_eq!(ratio_riccati_k(10, 5.0, 10.0), 5633.13, 1e-5);
    }
}
