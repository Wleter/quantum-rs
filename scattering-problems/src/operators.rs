#[macro_export]
macro_rules! get_spin_rot {
    ($basis:expr, $rot:path, $spin:path, $gamma:expr) => {
        Operator::from_mel(
            $basis, 
            [$rot((0, 0, 0)), $spin(half_u32!(0))],
            |[ang_braket, s_braket]| {
                let (l_ket, j_ket, j_tot_ket) = quantum::cast_variant!(ang_braket.ket.0, $rot);
                let m_r_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = quantum::cast_variant!(ang_braket.bra.0, $rot);
                let m_r_bra = ang_braket.bra.1;

                let s_ket = quantum::cast_variant!(s_braket.ket.0, $spin);
                let ms_ket = s_braket.ket.1;
                let s_bra = quantum::cast_variant!(s_braket.bra.0, $spin);
                let ms_bra = s_braket.bra.1;

                if l_bra == l_ket && j_bra == j_ket && s_ket == s_bra {
                    let factor = (((2 * j_tot_ket + 1) * (2 * j_tot_bra + 1) 
                                * (2 * j_ket + 1) * j_ket * (j_ket + 1)) as f64).sqrt()
                            * ((2. * s_ket.value() + 1.) * s_ket.value() * (s_ket.value() + 1.)).sqrt();

                    let sign = (-1.0f64).powi(1 + (j_tot_bra + j_tot_ket + l_bra + j_bra) as i32 - m_r_bra.double_value() / 2
                                                + (s_bra.double_value() as i32 - ms_bra.double_value()) / 2);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let j_tot_ket = HalfU32::from_doubled(2 * j_tot_ket);
                    
                    let mut wigner_sum = 0.;
                    for p in [half_i32!(-1), half_i32!(0), half_i32!(1)] { 
                        wigner_sum += (-1.0f64).powi(p.double_value() / 2) 
                            * wigner_6j(j_bra, j_tot_bra, l_bra, j_tot_ket, j_bra, half_u32!(1))
                            * wigner_3j(j_tot_bra, half_u32!(1), j_tot_ket, -m_r_bra, p, m_r_ket)
                            * wigner_3j(s_bra, half_u32!(1), s_bra, -ms_bra, -p, ms_ket)
                    }

                    $gamma * factor * sign * wigner_sum
                } else {
                    0.
                }
            }
        )
    };
}

#[macro_export]
macro_rules! get_aniso_hifi {
    ($basis:expr, $rot:path, $spin_s:path, $spin_i:path, $coupling:expr) => {
        Operator::from_mel(
            $basis, 
            [$rot((0, 0, 0)), $spin_s(half_u32!(0)), $spin_i(half_u32!(0))],
            |[ang_braket, s_braket, i_braket]| {
                let (l_ket, j_ket, j_tot_ket) = quantum::cast_variant!(ang_braket.ket.0, $rot);
                let mr_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = quantum::cast_variant!(ang_braket.bra.0, $rot);
                let mr_bra = ang_braket.bra.1;

                let s_ket = quantum::cast_variant!(s_braket.ket.0, $spin_s);
                let ms_ket = s_braket.ket.1;
                let s_bra = quantum::cast_variant!(s_braket.bra.0, $spin_s);
                let ms_bra = s_braket.bra.1;

                let i_ket = quantum::cast_variant!(i_braket.ket.0, $spin_i);
                let mi_ket = i_braket.ket.1;
                let i_bra = quantum::cast_variant!(i_braket.bra.0, $spin_i);
                let mi_bra = i_braket.bra.1;

                if l_bra == l_ket && s_ket == s_bra && i_ket == i_bra {
                    let factor = (((2 * j_tot_ket + 1) * (2 * j_tot_bra + 1)
                                    * (2 * j_ket + 1) * (2 * j_bra + 1)) as f64).sqrt()
                                * ((2. * s_bra.value() + 1.) * s_bra.value() * (s_bra.value() + 1.)
                                    * (2. * i_bra.value() + 1.) * i_bra.value() * (i_bra.value() + 1.)).sqrt();

                    let sign = (-1.0f64).powi((j_tot_bra + j_tot_ket + l_bra) as i32 - mr_bra.double_value() / 2);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let j_ket = HalfU32::from_doubled(2 * j_ket);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let j_tot_ket = HalfU32::from_doubled(2 * j_tot_ket);

                    let wigner = wigner_6j(j_bra, j_tot_bra, l_bra, j_tot_ket, j_ket, half_u32!(2))
                        * wigner_3j(half_u32!(1), half_u32!(1), half_u32!(2), mi_bra - mi_ket, ms_bra - ms_ket, mr_bra - mr_ket)
                        * wigner_3j(j_tot_bra, half_u32!(2), j_tot_ket, -mr_bra, mr_bra - mr_ket, mr_ket)
                        * wigner_3j(j_bra, half_u32!(2), j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                        * wigner_3j(i_bra, half_u32!(1), i_bra, -mi_bra, mi_bra - mi_ket, mi_ket)
                        * wigner_3j(s_bra, half_u32!(1), s_bra, -ms_bra, ms_bra - ms_ket, ms_ket);

                    $coupling * f64::sqrt(30.) / 3. * sign * factor * wigner
                } else {
                    0.
                }
            }
        )
    };
}


#[macro_export]
macro_rules! get_rotor_atom_potential_masking {
    (Singlet $lambda:expr; $basis:expr, $rot:path, $rotor_s:path, $atom_s:path) => {
        Operator::from_mel(
            $basis, 
            [$rot((0, 0, 0)), $rotor_s(half_u32!(0)), $atom_s(half_u32!(0))],
            |[ang_braket, s_braket, sa_braket]| {
                let (l_ket, j_ket, j_tot_ket) = quantum::cast_variant!(ang_braket.ket.0, $rot);
                let mr_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = quantum::cast_variant!(ang_braket.bra.0, $rot);
                let mr_bra = ang_braket.bra.1;

                let s_ket = quantum::cast_variant!(s_braket.ket.0, $rotor_s);
                let ms_ket = s_braket.ket.1;
                let s_bra = quantum::cast_variant!(s_braket.bra.0, $rotor_s);
                let ms_bra = s_braket.bra.1;

                let sa_ket = quantum::cast_variant!(sa_braket.ket.0, $atom_s);
                let msa_ket = sa_braket.ket.1;
                let sa_bra = quantum::cast_variant!(sa_braket.bra.0, $atom_s);
                let msa_bra = sa_braket.bra.1;

                if j_tot_bra == j_tot_ket && mr_bra == mr_ket && s_ket == s_bra && sa_ket == sa_bra {
                    let factor = (((2 * j_bra + 1) * (2 * j_ket + 1)
                                * (2 * l_bra + 1) * (2 * l_ket + 1)) as f64).sqrt();

                    let sign = (-1.0f64).powi((j_tot_bra + j_bra + j_ket + s_bra.double_value()) as i32 - sa_bra.double_value() as i32);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let j_ket = HalfU32::from_doubled(2 * j_ket);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let l_ket = HalfU32::from_doubled(2 * l_ket);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let lambda = HalfU32::from_doubled(2 * $lambda);

                    let wigner = wigner_6j(j_bra, l_bra, j_tot_bra, l_ket, j_ket, lambda)
                        * wigner_3j(j_bra, lambda, j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                        * wigner_3j(l_bra, lambda, l_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                        * wigner_3j(s_bra, sa_bra, half_u32!(0), ms_bra, msa_bra, half_i32!(0))
                        * wigner_3j(s_bra, sa_bra, half_u32!(0), ms_ket, msa_ket, half_i32!(0));

                    sign * factor * wigner
                } else {
                    0.
                }
            }
        )
    };
    (Triplet $lambda:expr; $basis:expr, $rot:path, $rotor_s:path, $atom_s:path) => {
        Operator::from_mel(
            $basis, 
            [$rot((0, 0, 0)), $rotor_s(half_u32!(0)), $atom_s(half_u32!(0))],
            |[ang_braket, s_braket, sa_braket]| {
                let (l_ket, j_ket, j_tot_ket) = quantum::cast_variant!(ang_braket.ket.0, $rot);
                let mr_ket = ang_braket.ket.1;

                let (l_bra, j_bra, j_tot_bra) = quantum::cast_variant!(ang_braket.bra.0, $rot);
                let mr_bra = ang_braket.bra.1;

                let s_ket = quantum::cast_variant!(s_braket.ket.0, $rotor_s);
                let ms_ket = s_braket.ket.1;
                let s_bra = quantum::cast_variant!(s_braket.bra.0, $rotor_s);
                let ms_bra = s_braket.bra.1;

                let sa_ket = quantum::cast_variant!(sa_braket.ket.0, $atom_s);
                let msa_ket = sa_braket.ket.1;
                let sa_bra = quantum::cast_variant!(sa_braket.bra.0, $atom_s);
                let msa_bra = sa_braket.bra.1;

                if j_tot_bra == j_tot_ket && mr_bra == mr_ket && s_ket == s_bra && sa_ket == sa_bra {
                    let factor = (((2 * j_bra + 1) * (2 * j_ket + 1)
                                * (2 * l_bra + 1) * (2 * l_ket + 1)) as f64).sqrt();

                    let sign = (-1.0f64).powi((j_tot_bra + j_bra + j_ket + s_bra.double_value()) as i32 - sa_bra.double_value() as i32);

                    let j_bra = HalfU32::from_doubled(2 * j_bra);
                    let j_ket = HalfU32::from_doubled(2 * j_ket);
                    let l_bra = HalfU32::from_doubled(2 * l_bra);
                    let l_ket = HalfU32::from_doubled(2 * l_ket);
                    let j_tot_bra = HalfU32::from_doubled(2 * j_tot_bra);
                    let lambda = HalfU32::from_doubled(2 * $lambda);

                    let mut triplet_wigner = 0.;
                    for ms_tot in [half_i32!(-1), half_i32!(0), half_i32!(1)] {
                       triplet_wigner += 3. * (-1.0f64).powi(ms_tot.double_value()) 
                        * wigner_3j(s_bra, sa_bra, half_u32!(1), ms_bra, msa_bra, -ms_tot)
                        * wigner_3j(s_bra, sa_bra, half_u32!(1), ms_ket, msa_ket, -ms_tot)
                    }

                    let wigner = wigner_6j(j_bra, l_bra, j_tot_bra, l_ket, j_ket, lambda)
                        * wigner_3j(j_bra, lambda, j_ket, half_i32!(0), half_i32!(0), half_i32!(0))
                        * wigner_3j(l_bra, lambda, l_ket, half_i32!(0), half_i32!(0), half_i32!(0));

                    sign * factor * wigner * triplet_wigner
                } else {
                    0.
                }
            }
        )
    }
}
