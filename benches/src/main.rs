
#[cfg(feature = "allocations")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "allocations")]
    piv_lu();

    #[cfg(feature = "allocations")]
    ldlt();
}

#[cfg(feature = "allocations")]
fn piv_lu() {
    use benches::setup;
    use scattering_solver::utility::inverse_inplace;

    let _profiler = dhat::Profiler::new_heap();

    let (mat, mut out, mut perm, mut perm_inv) = setup(500);

    inverse_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
}

#[cfg(feature = "allocations")]
fn ldlt() {
    use benches::setup;
    use scattering_solver::utility::inverse_symmetric_inplace;

    let _profiler = dhat::Profiler::new_heap();

    let (mat, mut out, mut perm, mut perm_inv) = setup(500);

    inverse_symmetric_inplace(mat.as_ref(), out.as_mut(), &mut perm, &mut perm_inv)
}