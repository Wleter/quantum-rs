use clebsch_gordan::{hi32, half_integer::HalfU32, wigner_3j, wigner_6j};

pub fn percival_coef(lambda: u32, lj_left: (HalfU32, HalfU32), lj_right: (HalfU32, HalfU32), j_tot: HalfU32) -> f64 {
    let lambda = HalfU32::from_doubled(2 * lambda);
    let l_left = lj_left.0;
    let j_left = lj_left.1;

    let l_right = lj_right.0;
    let j_right = lj_right.1;

    let mut wigners = wigner_3j(l_left, lambda, l_right, hi32!(0), hi32!(0), hi32!(0));
    if wigners == 0. { return 0. }

    wigners *= wigner_3j(j_left, lambda, j_right, hi32!(0), hi32!(0), hi32!(0));
    if wigners == 0. { return 0. }

    wigners *= wigner_6j(l_left, lambda, l_right, j_right, j_tot, j_left);
    if wigners == 0. { return 0. }
    
    let sign = (-1.0f64).powi(((l_left + l_right).double_value() as i32 - j_tot.double_value() as i32) / 2);
    let prefactor = ((2. * j_left.value() + 1.) * (2. * j_right.value() + 1.) 
                        * (2. * l_left.value() + 1.) * (2. * l_right.value() + 1.)).sqrt();

    sign * prefactor * wigners
}

#[derive(Clone, Copy, Debug)]
pub struct RotorLMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorJMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorJTot(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorJTotMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct GammaSpinRot(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct AnisoHifi(pub f64);

#[macro_export]
macro_rules! cast_spin_braket {
    ($braket:expr, $pat:path) => {{
        let s_bra = quantum::cast_variant!($braket.bra.0, $pat);
        let s_bra = quantum::states::spins::Spin::new(s_bra, $braket.bra.1);

        let s_ket = quantum::cast_variant!($braket.ket.0, $pat);
        let s_ket = quantum::states::spins::Spin::new(s_ket, $braket.ket.1);

        (s_bra, s_ket)
    }};
}