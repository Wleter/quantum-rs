use clebsch_gordan::{half_integer::HalfU32, wigner_3j, wigner_6j};

pub fn percival_coef(lambda: u32, dlj_left: (HalfU32, HalfU32), dlj_right: (HalfU32, HalfU32), dj_tot: HalfU32) -> f64 {
    let double_lambda = 2 * lambda;
    let dl_left = dlj_left.0;
    let dj_left = dlj_left.1;

    let dl_right = dlj_right.0;
    let dj_right = dlj_right.1;

    let mut wigners = wigner_3j(dl_left, double_lambda, dl_right, 0, 0, 0);
    if wigners == 0. { return 0. }

    wigners *= wigner_3j(dj_left, double_lambda, dj_right, 0, 0, 0);
    if wigners == 0. { return 0. }

    wigners *= wigner_6j(dl_left, double_lambda, dl_right, dj_right, dj_tot, dj_left);
    if wigners == 0. { return 0. }
    
    let sign = (-1.0f64).powi(((dl_left + dl_right) as i32 - dj_tot as i32) / 2);
    let prefactor = (((dj_left + 1) * (dj_right + 1) 
                        * (dl_left + 1) * (dl_right + 1)) as f64).sqrt();

    sign * prefactor * wigners
}

#[derive(Clone, Copy, Debug)]
pub struct RotorDoubleLMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorDoubleJMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorDoubleJTot(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorDoubleJTotMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct GammaSpinRot(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct AnisoHifi(pub f64);