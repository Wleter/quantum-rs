use diol::prelude::*;
use rand::Rng;
use scattering_solver::faer::Mat;
use scattering_solver::utility::{get_inverse_buffer, get_symmetric_inverse_buffer, inverse_inplace, inverse_symmetric_inplace};

fn main() -> eyre::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register(bench_inverse_piv_lu, [4, 32, 128, 512, 1024]);
    bench.register(bench_inverse_ldlt, [4, 32, 128, 512, 1024]);

    bench.run()?;
    Ok(())
}

pub fn setup(size: usize) -> (Mat<f64>, Mat<f64>) {
    let mut rng = rand::rng();
    let mat = Mat::<f64>::from_fn(size, size, |_, _| rng.random_range(-1.0..=1.0));

    let out = Mat::zeros(size, size);

    (mat, out)
}

fn bench_inverse_piv_lu(b: Bencher, size: usize) {
    let (mat, mut out) = setup(size);
    let mut buffer = get_inverse_buffer(size);

    b.bench(|| {
        inverse_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        black_box(&mut out);
        black_box(&mut buffer);
    });
}

fn bench_inverse_ldlt(b: Bencher, size: usize) {
    let (mat, mut out) = setup(size);
    let mut buffer = get_symmetric_inverse_buffer(size);

    b.bench(|| {
        inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        black_box(&mut out);
        black_box(&mut buffer);
    });
}
