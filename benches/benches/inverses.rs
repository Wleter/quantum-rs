use diol::prelude::*;
use rand::Rng;
use scattering_solver::faer::Mat;
use scattering_solver::utility::{
    get_lblt_inverse_buffer, get_ldlt_inverse_buffer, get_lu_inverse_buffer, inverse_lblt_inplace,
    inverse_ldlt_inplace, inverse_lu_inplace,
};

fn main() -> eyre::Result<()> {
    let bench = Bench::new(Config::from_args()?);

    bench.register(
        "piv_lu inverse",
        bench_inverse_piv_lu,
        [4, 32, 128, 512, 1024],
    );
    bench.register("ldlt inverse", bench_inverse_ldlt, [4, 32, 128, 512, 1024]);
    bench.register("lblt inverse", bench_inverse_lblt, [4, 32, 128, 512, 1024]);

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
    let mut buffer = get_lu_inverse_buffer(size);

    b.bench(|| {
        inverse_lu_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        black_box(&mut out);
        black_box(&mut buffer);
    });
}

fn bench_inverse_ldlt(b: Bencher, size: usize) {
    let (mat, mut out) = setup(size);
    let mut buffer = get_ldlt_inverse_buffer(size);

    b.bench(|| {
        inverse_ldlt_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        black_box(&mut out);
        black_box(&mut buffer);
    });
}

fn bench_inverse_lblt(b: Bencher, size: usize) {
    let (mat, mut out) = setup(size);
    let mut buffer = get_lblt_inverse_buffer(size);

    b.bench(|| {
        inverse_lblt_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        black_box(&mut out);
        black_box(&mut buffer);
    });
}
