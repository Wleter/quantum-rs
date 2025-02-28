use clebsch_gordan::{
    half_integer::{HalfI32, HalfU32},
    hi32, hu32, wigner_3j, wigner_6j,
};
use quantum::states::{
    braket::Braket,
    spins::{Spin, SpinOperators},
};

use crate::alkali_rotor_atom::ParityBlock;

#[derive(Clone, Copy, Debug)]
pub struct GammaSpinRot(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct AnisoHifi(pub f64);

#[derive(Clone, Copy, PartialEq, Hash, Default)]
pub struct AngularPair {
    pub l: HalfU32,
    pub n: HalfU32,
}

impl AngularPair {
    pub fn new(l: HalfU32, n: HalfU32) -> Self {
        Self { l, n }
    }
}

impl std::fmt::Debug for AngularPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.l, self.n)
    }
}

pub fn create_angular_pairs(
    l_max: u32,
    n_max: u32,
    n_tot_max: u32,
    parity: ParityBlock,
) -> Vec<AngularPair> {
    let ls: Vec<u32> = (0..=l_max).collect();

    let mut angular_states = vec![];
    for &l in &ls {
        let n_lower = ((l as i32 - n_tot_max as i32).max(0)).unsigned_abs();
        let n_upper = (l + n_tot_max).min(n_max);

        for n in n_lower..=n_upper {
            match parity {
                ParityBlock::Positive => {
                    if (n + l) & 1 == 1 {
                        continue;
                    }
                }
                ParityBlock::Negative => {
                    if (n + l) & 1 == 0 {
                        continue;
                    }
                }
                ParityBlock::All => (),
            }

            angular_states.push(AngularPair {
                l: l.into(),
                n: n.into(),
            });
        }
    }

    angular_states
}

#[rustfmt::skip]
pub fn percival_coef(lambda: u32, ang: Braket<AngularPair>, n_tot: HalfU32) -> f64 {
    let lambda = HalfU32::from_doubled(2 * lambda);

    let sign = (-1.0f64).powi(((ang.bra.l + ang.ket.l).double_value() as i32 
        - n_tot.double_value() as i32) / 2);
        
    let prefactor = p1_factor(ang.bra.n) * p1_factor(ang.ket.n)
        * p1_factor(ang.bra.l) * p1_factor(ang.ket.l);

    let wigners = wigner_3j(ang.bra.l, lambda, ang.ket.l, hi32!(0), hi32!(0), hi32!(0))
        * wigner_3j(ang.bra.n, lambda, ang.ket.n, hi32!(0), hi32!(0), hi32!(0))
        * wigner_6j(ang.bra.l, lambda, ang.ket.l, ang.ket.n, n_tot, ang.bra.n);

    sign * prefactor * wigners
}

#[rustfmt::skip]
pub fn percival_coef_tram_mel(lambda: u32, ang: Braket<AngularPair>, n_tot: Braket<Spin>) -> f64 {
    if n_tot.bra == n_tot.ket {
        let lambda = HalfU32::from_doubled(2 * lambda);

        let sign = (-1.0f64).powi(((ang.bra.l + ang.ket.l).double_value() as i32 
            - n_tot.bra.s.double_value() as i32) / 2);
            
        let prefactor = p1_factor(ang.bra.n) * p1_factor(ang.ket.n)
            * p1_factor(ang.bra.l) * p1_factor(ang.ket.l);
    
        let wigners = wigner_3j(ang.bra.l, lambda, ang.ket.l, hi32!(0), hi32!(0), hi32!(0))
            * wigner_3j(ang.bra.n, lambda, ang.ket.n, hi32!(0), hi32!(0), hi32!(0))
            * wigner_6j(ang.bra.l, lambda, ang.ket.l, ang.ket.n, n_tot.bra.s, ang.bra.n);
    
        sign * prefactor * wigners
    } else {
        0.0
    }
}

#[rustfmt::skip]
pub fn singlet_projection_uncoupled(s1: Braket<Spin>, s2: Braket<Spin>) -> f64 {
    let singlet_spin = Spin::zero();
    SpinOperators::clebsch_gordan(s1.bra, s2.bra, singlet_spin) 
        * SpinOperators::clebsch_gordan(s1.ket, s2.ket, singlet_spin)
}

#[rustfmt::skip]
pub fn triplet_projection_uncoupled(s1: Braket<Spin>, s2: Braket<Spin>) -> f64 {
    let mut value = 0.0;

    for ms in [hi32!(-1), hi32!(0), hi32!(1)] {
        let triplet_spin = Spin::new(hu32!(1), ms);
        value += SpinOperators::clebsch_gordan(s1.bra, s2.bra, triplet_spin) 
            * SpinOperators::clebsch_gordan(s1.ket, s2.ket, triplet_spin)
    }

    value
}

#[rustfmt::skip]
pub fn spin_rot_tram_mel(ang: Braket<AngularPair>, n_tot: Braket<Spin>, s: Braket<Spin>) -> f64 {
    if ang.bra.l == ang.ket.l && ang.bra.n == ang.ket.n && s.bra.s == s.ket.s {
        let factor = p1_factor(n_tot.ket.s) * p1_factor(n_tot.bra.s)
            * p3_factor(ang.bra.n) 
            * p3_factor(s.bra.s);

        let sign = (-1f64).powi(1 + (ang.bra.n + ang.bra.l + n_tot.ket.s).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s.bra);

        let mut wigner_sum = 0.;
        for p in [-1, 0, 1] { 
            wigner_sum += (-1.0f64).powi(p) 
                * wigner_6j(ang.bra.n, hu32!(1), ang.bra.n, n_tot.bra.s, ang.bra.l, n_tot.ket.s)
                * wigner_3j(n_tot.bra.s, hu32!(1), n_tot.ket.s, -n_tot.bra.ms, p.into(), n_tot.ket.ms)
                * wigner_3j(s.bra.s, hu32!(1), s.bra.s, -s.bra.ms, (-p).into(), s.ket.ms)
        }

        factor * sign * wigner_sum
    } else {
        0.
    }
}

#[rustfmt::skip]
pub fn aniso_hifi_tram_mel(ang: Braket<AngularPair>, n_tot: Braket<Spin>, s: Braket<Spin>, i: Braket<Spin>) -> f64 {
    if ang.bra.l == ang.ket.l && s.bra.s == s.ket.s && i.bra.s == i.ket.s {
        let factor = p1_factor(n_tot.ket.s) * p1_factor(n_tot.bra.s)
            * p1_factor(ang.bra.n) * p1_factor(ang.ket.n)
            * p3_factor(i.bra.s) 
            * p3_factor(s.bra.s);

        let sign = (-1f64).powi((ang.bra.l + n_tot.ket.s).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s.bra)
            * spin_phase_factor(i.bra);


        let wigner = clebsch_gordan::wigner_6j(ang.bra.n, hu32!(2), ang.ket.n, n_tot.ket.s, ang.bra.l, n_tot.bra.s)
            * clebsch_gordan::wigner_3j(hu32!(1), hu32!(1), hu32!(2), 
                                        i.bra.ms - i.ket.ms, s.bra.ms - s.ket.ms, n_tot.bra.ms - n_tot.ket.ms)
            * clebsch_gordan::wigner_3j(n_tot.bra.s, hu32!(2), n_tot.ket.s, 
                                        -n_tot.bra.ms, n_tot.bra.ms - n_tot.ket.ms, n_tot.ket.ms)
            * clebsch_gordan::wigner_3j(ang.bra.n, hu32!(2), ang.ket.n, hi32!(0), hi32!(0), hi32!(0))
            * clebsch_gordan::wigner_3j(i.bra.s, hu32!(1), i.bra.s, -i.bra.ms, i.bra.ms - i.ket.ms, i.ket.ms)
            * clebsch_gordan::wigner_3j(s.bra.s, hu32!(1), s.bra.s, -s.bra.ms, s.bra.ms - s.ket.ms, s.ket.ms);

        f64::sqrt(30.) / 3. * sign * factor * wigner
    } else {
        0.
    }
}

#[rustfmt::skip]
pub fn dipole_dipole_tram_mel(ang: Braket<AngularPair>, n_tot: Braket<Spin>, s_r: Braket<Spin>, s_a: Braket<Spin>) -> f64 {
    if ang.bra.n == ang.ket.n && s_r.bra.s == s_r.ket.s && s_a.bra.s == s_a.ket.s {
        let factor = p1_factor(n_tot.bra.s) * p1_factor(n_tot.ket.s)
            * p1_factor(ang.bra.l) * p1_factor(ang.ket.l)
            * p3_factor(s_r.bra.s) * p3_factor(s_a.bra.s);

        let sign = (-1f64).powi((n_tot.bra.s + ang.bra.l + ang.ket.l + ang.bra.n).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s_r.bra) 
            * spin_phase_factor(s_a.bra);


        let wigner = clebsch_gordan::wigner_6j(ang.bra.l, hu32!(2), ang.ket.l, n_tot.ket.s, ang.bra.n, n_tot.bra.s)
            * clebsch_gordan::wigner_3j(hu32!(1), hu32!(1), hu32!(2), 
                                        s_r.bra.ms - s_r.ket.ms, s_a.bra.ms - s_a.ket.ms, n_tot.bra.ms - n_tot.ket.ms)
            * clebsch_gordan::wigner_3j(ang.bra.l, hu32!(2), ang.ket.l, hi32!(0), hi32!(0), hi32!(0))
            * clebsch_gordan::wigner_3j(n_tot.bra.s, hu32!(2), n_tot.bra.s, -n_tot.bra.ms, n_tot.bra.ms - n_tot.ket.ms, n_tot.ket.ms)
            * clebsch_gordan::wigner_3j(s_a.bra.s, hu32!(1), s_a.bra.s, -s_a.bra.ms, s_a.bra.ms - s_a.ket.ms, s_a.ket.ms)
            * clebsch_gordan::wigner_3j(s_r.bra.s, hu32!(1), s_r.bra.s, -s_r.bra.ms, s_r.bra.ms - s_r.ket.ms, s_r.ket.ms);

        -f64::sqrt(30.) * sign * factor * wigner
    } else {
        0.
    }
}

#[rustfmt::skip]
pub fn percival_coef_uncoupled_mel(lambda: u32, n: Braket<Spin>, l: Braket<Spin>) -> f64 {    
    let factor = p1_factor(n.bra.s) * p1_factor(n.ket.s)
        * p1_factor(l.bra.s) * p1_factor(l.ket.s);

    let wigners = wigner_3j(l.bra.s, lambda.into(), l.ket.s, hi32!(0), hi32!(0), hi32!(0))
        * wigner_3j(n.bra.s, lambda.into(), n.ket.s, hi32!(0), hi32!(0), hi32!(0));

    let summed = (0..=lambda)
        .map(|m_lambda| {
            let lambda: HalfU32 = lambda.into();
            let m_lambda: HalfI32 = (m_lambda as i32).into();

            (-1.0f64).powf((m_lambda - l.bra.ms - n.bra.ms).value())
                * wigner_3j(l.bra.s, lambda, l.ket.s, -l.bra.ms, -m_lambda, l.ket.ms)
                * wigner_3j(n.bra.s, lambda, n.ket.s, -n.bra.ms, m_lambda, n.ket.ms)
        })
        .sum::<f64>();

    factor * wigners * summed
}

#[rustfmt::skip]
pub fn aniso_hifi_uncoupled_mel(n: Braket<Spin>, s: Braket<Spin>, i: Braket<Spin>) -> f64 {
    let factor = f64::sqrt(30.) / 3. * p3_factor(s.bra.s) * p3_factor(i.bra.s) 
        * p1_factor(n.bra.s) * p1_factor(n.ket.s);

    let sign = (-1f64).powi(-n.bra.ms.double_value() / 2) * spin_phase_factor(s.bra) * spin_phase_factor(i.bra);

    let wigners = wigner_3j(n.bra.s, hu32!(2), n.ket.s, -n.bra.ms, n.bra.ms - n.ket.ms, n.ket.ms)
        * wigner_3j(n.bra.s, hu32!(2), n.ket.s, hi32!(0), hi32!(0), hi32!(0))
        * wigner_3j(hu32!(1), hu32!(1), hu32!(2), s.bra.ms - s.ket.ms, i.bra.ms - i.ket.ms, n.bra.ms - n.ket.ms)
        * wigner_3j(s.bra.s, hu32!(1), s.ket.s, -s.bra.ms, s.bra.ms - s.ket.ms, s.ket.ms)
        * wigner_3j(i.bra.s, hu32!(1), i.ket.s, -i.bra.ms, i.bra.ms - i.ket.ms, i.ket.ms);

    factor * sign * wigners
}

#[inline]
/// Calculates sqrt(2s + 1)
pub fn p1_factor(s: HalfU32) -> f64 {
    (2. * s.value() + 1.).sqrt()
}

#[inline]
/// Calculates sqrt((2s + 1)s(s + 1))
pub fn p3_factor(s: HalfU32) -> f64 {
    let s = s.value();
    ((2. * s + 1.) * s * (s + 1.)).sqrt()
}

#[inline]
/// Calculates (-1)^(s - ms)
pub fn spin_phase_factor(s: Spin) -> f64 {
    (-1.0f64).powi((s.s.double_value() as i32 - s.ms.double_value()) / 2)
}
