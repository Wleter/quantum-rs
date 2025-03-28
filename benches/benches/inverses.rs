use scattering_solver::faer::Mat;
use scattering_solver::utility::{inverse_inplace, inverse_symmetric_inplace};
use rand::Rng;
use diol::prelude::*;

fn main() -> eyre::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register(bench_inverse_piv_lu, [4, 32, 128, 512, 1024]);
    bench.register(bench_inverse_lblt, [4, 32, 128, 512, 1024]);

    bench.run()?;
    Ok(())
}

pub fn setup(size: usize) -> (Mat<f64>, Mat<f64>, Vec<usize>, Vec<usize>) {
    let mut rng = rand::rng();
    let mat = Mat::<f64>::from_fn(size, size, |_, _| rng.random_range(-1.0..=1.0));
    
    let out = Mat::zeros(size, size);
    let perm = vec![0; size];
    let perm_inv = vec![0; size];

    (mat, out, perm, perm_inv)
}

fn bench_inverse_piv_lu(b: Bencher, size: usize) {
    let (mat, mut out, mut perm, mut perm_inv) = setup(size);

    b.bench(|| {
        inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv);
        
        black_box(&mut out);
        black_box(&mut perm);
        black_box(&mut perm_inv);
    });
}

fn bench_inverse_lblt(b: Bencher, size: usize) {
    let (mat, mut out, mut perm, mut perm_inv) = setup(size);

    b.bench(|| {
        inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv);

        black_box(&mut out);
        black_box(&mut perm);
        black_box(&mut perm_inv);
    });
}
