use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::Path,
};

use faer::{Mat, MatRef, Side};

pub fn spin_proj(dspin: u32) -> Vec<i32> {
    (-(dspin as i32)..=(dspin as i32)).step_by(2).collect()
}

pub fn sum_spin_proj(dspin1: u32, dspin2: u32) -> Vec<(u32, Vec<i32>)> {
    let dspin_max = dspin1 + dspin2;
    let dspin_min = (dspin1 as i32 - dspin2 as i32).unsigned_abs();

    (dspin_min..=dspin_max)
        .step_by(2)
        .map(|s| (s, spin_proj(s)))
        .collect()
}

pub fn diagonalize(mat: MatRef<f64>) -> (Vec<f64>, Mat<f64>) {
    let eigen = mat.selfadjoint_eigendecomposition(Side::Upper);
    let values = eigen.s().column_vector().try_as_slice().unwrap().into();

    (values, eigen.u().to_owned())
}

pub fn save_spectrum(
    header: &str,
    filename: &str,
    parameter: &[f64],
    energies: &[Vec<f64>],
) -> Result<(), std::io::Error> {
    assert_eq!(
        parameter.len(),
        energies.len(),
        "parameters and energies have to have the same length"
    );

    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("dat");
    let filepath = path.parent().unwrap();

    let mut buf = header.to_string();

    for (p, e) in parameter.iter().zip(energies.iter()) {
        let line = e
            .iter()
            .fold(format!("{:e}", p), |s, val| s + &format!("\t{:e}", val));

        buf.push_str(&format!("\n{line}"))
    }

    if !Path::new(filepath).exists() {
        create_dir_all(filepath)?;
        println!("created path {}", filepath.display());
    }

    let mut file = File::create(&path)?;
    file.write_all(buf.as_bytes())?;

    println!("saved data on {}", path.display());
    Ok(())
}

#[macro_export]
macro_rules! get_zeeman_prop {
    ($basis:expr, $state:path, $gamma:expr) => {{
        quantum::states::operator::Operator::from_mel(&($basis), [$state(0)], |[s]| {
            let s_bra = quantum::cast_variant!(s.bra.0, $state);
            let s_bra = quantum::states::spins::DoubleSpin(s_bra, s.bra.1);

            let s_ket = quantum::cast_variant!(s.ket.0, $state);
            let s_ket = quantum::states::spins::DoubleSpin(s_ket, s.ket.1);

            -$gamma * quantum::states::spins::SpinOperators::proj_z(s_bra, s_ket)
        })
    }};
}

#[macro_export]
macro_rules! get_hifi {
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(0), $state2(0)],
            |[s, i]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let s_bra = quantum::states::spins::DoubleSpin(s_bra, s.bra.1);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let s_ket = quantum::states::spins::DoubleSpin(s_ket, s.ket.1);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let i_bra = quantum::states::spins::DoubleSpin(i_bra, i.bra.1);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let i_ket = quantum::states::spins::DoubleSpin(i_ket, i.ket.1);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr, with $fc:path) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(0), $state2(0), $fc],
            |[s, i, _]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let s_bra = quantum::states::spins::DoubleSpin(s_bra, s.bra.1);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let s_ket = quantum::states::spins::DoubleSpin(s_ket, s.ket.1);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let i_bra = quantum::states::spins::DoubleSpin(i_bra, i.bra.1);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let i_ket = quantum::states::spins::DoubleSpin(i_ket, i.ket.1);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
}
