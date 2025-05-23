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
        return vec![10.0f64.powf(start)];
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

/// Returns normalization factor of Associated Legendre Polynomial P_l^m.
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

/// Calculates riccati bessel function of the first kind j_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) =
        bessel_recurrence_deriv(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x));

    (value, value_deriv + value / x)
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) =
        bessel_recurrence_deriv(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x));

    (value, value_deriv + value / x)
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

/// Calculates the ratio of derivative and the value of the  
/// riccati modified spherical bessel function of the first kind
/// (that is $sqrt(x) I_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.2 (written as z * sqrt(pi/2z) I_{n+1/2}(z))
pub fn ratio_riccati_i_deriv(n: u32, x: f64) -> f64 {
    let red_i_0 = (1. - f64::exp(-2.0 * x)) / 2.0;
    let red_i_1 = -(1. - f64::exp(-2.0 * x)) / (2.0 * x) + (1. + f64::exp(-2.0 * x)) / 2.0;

    // Calculates riccati I bessel, without leading exponent, and its derivative
    let (i_red, i_red_deriv) = modified_bessel_recurrence_deriv(n, x, red_i_0, red_i_1);

    i_red_deriv / i_red + 1.0 / x
}

/// Calculates the ratio of derivative and the value of the  
/// riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn ratio_riccati_k_deriv(n: u32, x: f64) -> f64 {
    let red_k_0 = 1.0;
    let red_k_1 = 1.0 + 1.0 / x;

    // Calculates riccati $(-1)^(n+1) * K$ bessel without leading exponent
    let (k_red, k_red_deriv) = modified_bessel_recurrence_deriv(n, x, -red_k_0, red_k_1);

    k_red_deriv / k_red + 1.0 / x
}

/// Calculates f_n(x) given n, x, f_0(x), f_1(x)
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

/// Calculates f_n(x) and its derivative (g(x) d Bessel(x)/dx) given n, x, f_0(x), f_1(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19
fn bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
    if n == 0 {
        return (f_0, -f_1);
    }
    if n == 1 {
        return (f_1, f_0 - (n + 1) as f64 / x * f_1);
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }

    (f_k, f_k_1 - (n + 1) as f64 / x * f_k)
}

/// Calculates f_n(x) given n, x, f_0(x), f_1(x).
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

/// Calculates f_n(x) and its derivative (g(x) d MBessel(x)/dx) given n, x, f_0(x), f_1(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn modified_bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
    if n == 0 {
        return (f_0, f_1);
    }
    if n == 1 {
        return (f_1, f_0 - (n + 1) as f64 / x * f_1);
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = f_k_1 - (2 * k + 1) as f64 / x * f_k;
        f_k_1 = f_k;
        f_k = f_new;
    }

    (f_k, f_k_1 - (n + 1) as f64 / x * f_k)
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
/// - `assert_approx_eq!(x, y, err)` for single element
/// - `assert_approx_eq!(iter => x, y, err)` for elements in slice
/// - `assert_approx_eq!(mat => x, y, err)` for elements in matrices
macro_rules! assert_approx_eq {
    ($x:expr, $y:expr, $err:expr $(, $message:expr)?) => {
        if ($x - $y).abs() > $x.abs() * $err {
            panic!(
                "assertion failed\nleft side: {:e}\nright side: {:e}",
                $x, $y
            )
        }
    };
    (iter => $x:expr, $y:expr, $err:expr) => {
        for (x, y) in $x.iter().zip(&$y) {
            assert_approx_eq!(x, y, $err);
        }
    };
    (mat => $x:expr, $y:expr, $err:expr) => {
        assert_eq!($x.nrows(), $y.nrows());
        assert_eq!($x.ncols(), $y.ncols());

        for i in 0..$x.nrows() {
            for j in 0..$x.ncols() {
                assert_approx_eq!($x[(i, j)], $y[(i, j)], $err);
            }
        }
    };
}

#[macro_export]
/// Check for approximate error |x - y| < x * err
///
/// # Syntax
///
/// - `approx_eq!(x, y, err)`
macro_rules! approx_eq {
    ($x:expr, $y:expr, $err:expr) => {
        ($x - $y).abs() < $x.abs() * $err
    };
}

#[cfg(test)]
mod test {
    use std::{
        thread::sleep,
        time::{Duration, Instant},
    };

    use crate::utility::{
        associated_legendre_polynomials, legendre_polynomials, logspace, ratio_riccati_i,
        ratio_riccati_i_deriv, ratio_riccati_k, ratio_riccati_k_deriv, riccati_j_deriv, riccati_n,
        riccati_n_deriv,
    };

    use super::{linspace, riccati_j};

    #[test]
    fn test_grids() {
        let grid = linspace(1.0, 15.0, 6);
        let expected = vec![1.0, 3.8, 6.6, 9.4, 12.2, 15.0];

        assert_approx_eq!(iter => grid, expected, 1e-6);

        let grid = logspace(-2.0, 3.0, 4);
        let expected = vec![0.01, 0.46415888, 21.5443469, 1000.0];

        assert_approx_eq!(iter => grid, expected, 1e-6);
    }

    #[test]
    fn test_legendre() {
        let legendre = legendre_polynomials(5, 0.3);
        let expected = vec![1.0, 0.3, -0.365, -0.3825, 0.0729375, 0.34538625];
        assert_approx_eq!(iter => legendre, expected, 1e-6);

        let legendre = legendre_polynomials(5, -0.7);
        let expected = vec![1.0, -0.7, 0.235, 0.1925, -0.4120625, 0.36519875];
        assert_approx_eq!(iter => legendre, expected, 1e-6);

        let legendre = associated_legendre_polynomials(5, 3, 0.3);
        let expected = vec![0.0, 0.0, 0.0, -13.02127, -27.3446672, 8.6591446];
        assert_approx_eq!(iter => legendre, expected, 1e-6);

        let legendre = associated_legendre_polynomials(5, 2, -0.7);
        let expected = vec![0.0, 0.0, 1.53, -5.355, 9.29475, -8.808975];
        assert_approx_eq!(iter => legendre, expected, 1e-6);
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

        assert_approx_eq!(riccati_j_deriv(5, 10.0).0, -0.555345, 1e-5);
        assert_approx_eq!(riccati_j_deriv(5, 10.0).1, -0.77822, 1e-5);
        assert_approx_eq!(riccati_j_deriv(10, 10.0).0, 0.646052, 1e-5);
        assert_approx_eq!(riccati_j_deriv(10, 10.0).1, 0.354913, 1e-5);

        assert_approx_eq!(riccati_n_deriv(5, 10.0).0, -0.938335, 1e-5);
        assert_approx_eq!(riccati_n_deriv(5, 10.0).1, 0.485767, 1e-5);
        assert_approx_eq!(riccati_n_deriv(10, 10.0).0, 1.72454, 1e-5);
        assert_approx_eq!(riccati_n_deriv(10, 10.0).1, -0.600479, 1e-5);

        assert_approx_eq!(ratio_riccati_i_deriv(5, 10.0), 1.1531, 1e-5);
        assert_approx_eq!(ratio_riccati_i_deriv(10, 10.0), 1.47691, 1e-5);

        assert_approx_eq!(ratio_riccati_k_deriv(5, 10.0), -1.12973, 1e-5);
        assert_approx_eq!(ratio_riccati_k_deriv(10, 10.0), -1.42441, 1e-5);
    }

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
}
