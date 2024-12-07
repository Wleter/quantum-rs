pub mod faer_backed;

// #[cfg(feature = "nalgebra")]
// pub mod smatrix_backed;

#[cfg(feature = "nalgebra")]
pub mod dmatrix_backed;

pub struct MultiRatioNumerovStep<T>
{
    f1: T,
    f2: T,
    f3: T,

    buffer1: T,
    buffer2: T,
}
