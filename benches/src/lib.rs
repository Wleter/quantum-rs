#![feature(test)]
#![allow(unused)]

use faer::Mat;

extern crate test;

fn setup(size: usize) -> (Mat<f64>, Mat<f64>, Vec<usize>, Vec<usize>) {
    let mat = Mat::from_fn(size, size, |i, j| {
        ((i + j) % (size / 4)) as f64 * 11. / 6.
    });
    let out = Mat::zeros(size, size);
    let perm = vec![0; size];
    let perm_inv = vec![0; size];

    (mat, out, perm, perm_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scattering_solver::utility::{inverse_inplace, inverse_symmetric_inplace};
    use test::Bencher;

    #[bench]
    fn bench_inverse_piv_lu_50(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(50);

        b.iter(|| {
            inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_piv_lu_100(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(100);

        b.iter(|| {
            inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_piv_lu_500(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(500);

        b.iter(|| {
            inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_piv_lu_2000(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(2000);

        b.iter(|| {
            inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_lblt_50(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(50);

        b.iter(|| {
            inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_lblt_100(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(100);

        b.iter(|| {
            inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_lblt_500(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(500);

        b.iter(|| {
            inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }

    #[bench]
    fn bench_inverse_lblt_2000(b: &mut Bencher) {
        let (mat, mut out, mut perm, mut perm_inv) = setup(2000);

        b.iter(|| {
            inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
        });
    }
}