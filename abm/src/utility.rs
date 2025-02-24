use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::Path,
};

use faer::{Mat, MatRef, Side};

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

/// Macro for generating an `Operator` for the proportional Zeeman effect.
///
/// This macro generates an operator for a proportional Zeeman effect by taking
/// a basis, a spin type, and gyrometric constant.
///
/// # Syntax
///
/// - `get_zeeman_prop!($basis, $state, $gamma)`
/// - `get_zeeman_prop!($basis, $state, $gamma; $vib)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$basis`: Basis on which the operator is defined.
/// - `$state`: Wanted spin state (e.g., a path to a specific enum variant).
/// - `$gamma`: The gyrometric constant for the Zeeman effect (`f64`).
///
/// ## Optional Arguments
/// - `$proj` (optional, second arm): The projection state value, used if basis have enum type values.
///
/// # Matched Arms
///
/// ## Arm 1: `get_zeeman_prop!($basis, $state, $gamma)`
/// Use this arm when the quantum system has i32 value type. The macro
/// constructs an operator where the Zeeman effect depends only on the spin states.
///
/// ## Arm 2: `get_zeeman_prop!($basis, $state, $gamma; $proj)`
/// Use this arm when the quantum system includes enum value type.
/// This arm constructs an operator that accounts for both spin and vibrational states.
#[macro_export]
macro_rules! get_zeeman_prop {
    ($basis:expr, $state:path, $gamma:expr) => {{
        quantum::states::operator::Operator::from_mel(&($basis), [$state(clebsch_gordan::hu32!(0))], |[s]| {
            let s_bra = quantum::cast_variant!(s.bra.0, $state);
            let s_bra = quantum::states::spins::Spin::new(s_bra, s.bra.1);

            let s_ket = quantum::cast_variant!(s.ket.0, $state);
            let s_ket = quantum::states::spins::Spin::new(s_ket, s.ket.1);

            -$gamma * quantum::states::spins::SpinOperators::proj_z(s_bra, s_ket)
        })
    }};
    ($basis:expr, $state:path, $gamma:expr; $proj:path) => {{
        quantum::states::operator::Operator::from_mel(&($basis), [$state(clebsch_gordan::hu32!(0))], |[s]| {
            let s_bra = quantum::cast_variant!(s.bra.0, $state);
            let ms_bra = quantum::cast_variant!(s.bra.1, $proj);
            let s_bra = quantum::states::spins::Spin::new(s_bra, ms_bra);

            let s_ket = quantum::cast_variant!(s.ket.0, $state);
            let ms_ket = quantum::cast_variant!(s.bra.1, $proj);
            let s_ket = quantum::states::spins::Spin::new(s_ket, ms_ket);

            -$gamma * quantum::states::spins::SpinOperators::proj_z(s_bra, s_ket)
        })
    }};
}

/// Macro for generating an `Operator` for the Hyperfine effect.
///
/// This macro generates an operator for a Hyperfine effect by taking
/// a quantum state basis, a state type, and gyrometric constant.
///
/// # Syntax
///
/// - `get_hifi!($basis, $state1, $state2, $a_hifi)`
/// - `get_hifi!($basis, $state1, $state2, $a_hifi; $fc)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$basis`: Basis on which the operator is defined.
/// - `$state1`: Wanted first spin state to couple to the second (e.g., a path to a specific enum variant).
/// - `$state2`: Wanted second spin state to couple to the first (e.g., a path to a specific enum variant).
/// - `$a_hifi`: hyperfine constant (`f64`).
///
/// ## Optional Arguments
/// - `$fc` (optional): Franck-Condon factors path in the basis.
/// - `$proj` (optional): The projection state value, used if basis have enum type values.
///
/// # Matched Arms
///
/// ## Arm 1: `get_hifi!($basis, $state1, $state2, $a_hifi)`
/// Use this arm when the quantum system has simple i32 value types and no 
/// Franck-Condon factors
///
/// ## Arm 2: `get_hifi!($basis, $state1, $state2, $a_hifi, with $fc)`
/// Use this arm when the quantum system has simple i32 value types and 
/// Franck-Condon factors.
#[macro_export]
macro_rules! get_hifi {
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(clebsch_gordan::hu32!(0)), $state2(clebsch_gordan::hu32!(0))],
            |[s, i]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let s_bra = quantum::states::spins::Spin::new(s_bra, s.bra.1);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let s_ket = quantum::states::spins::Spin::new(s_ket, s.ket.1);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let i_bra = quantum::states::spins::Spin::new(i_bra, i.bra.1);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let i_ket = quantum::states::spins::Spin::new(i_ket, i.ket.1);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr, with $fc:path) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(clebsch_gordan::hu32!(0)), $state2(clebsch_gordan::hu32!(0)), $fc],
            |[s, i, _]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let s_bra = quantum::states::spins::Spin::new(s_bra, s.bra.1);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let s_ket = quantum::states::spins::Spin::new(s_ket, s.ket.1);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let i_bra = quantum::states::spins::Spin::new(i_bra, i.bra.1);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let i_ket = quantum::states::spins::Spin::new(i_ket, i.ket.1);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr; $proj:path) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(clebsch_gordan::hu32!(0)), $state2(clebsch_gordan::hu32!(0))],
            |[s, i, _]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let ms_bra = quantum::cast_variant!(s.bra.1, $proj);
                let s_bra = quantum::states::spins::Spin::new(s_bra, ms_bra);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let ms_ket = quantum::cast_variant!(s.ket.1, $proj);
                let s_ket = quantum::states::spins::Spin::new(s_ket, ms_ket);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let mi_bra = quantum::cast_variant!(i.bra.1, $proj);
                let i_bra = quantum::states::spins::Spin::new(i_bra, mi_bra);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let mi_ket = quantum::cast_variant!(i.ket.1, $proj);
                let i_ket = quantum::states::spins::Spin::new(i_ket, mi_ket);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
    ($basis:expr, $state1:path, $state2:path, $a_hifi:expr, with $fc:path; $proj:path) => {{
        quantum::states::operator::Operator::from_mel(
            &($basis),
            [$state1(clebsch_gordan::hu32!(0)), $state2(clebsch_gordan::hu32!(0)), $fc],
            |[s, i, _]| {
                let s_bra = quantum::cast_variant!(s.bra.0, $state1);
                let ms_bra = quantum::cast_variant!(s.bra.1, $proj);
                let s_bra = quantum::states::spins::Spin::new(s_bra, ms_bra);

                let s_ket = quantum::cast_variant!(s.ket.0, $state1);
                let ms_ket = quantum::cast_variant!(s.ket.1, $proj);
                let s_ket = quantum::states::spins::Spin::new(s_ket, ms_ket);

                let i_bra = quantum::cast_variant!(i.bra.0, $state2);
                let mi_bra = quantum::cast_variant!(i.bra.1, $proj);
                let i_bra = quantum::states::spins::Spin::new(i_bra, mi_bra);

                let i_ket = quantum::cast_variant!(i.ket.0, $state2);
                let mi_ket = quantum::cast_variant!(i.ket.1, $proj);
                let i_ket = quantum::states::spins::Spin::new(i_ket, mi_ket);

                $a_hifi * quantum::states::spins::SpinOperators::dot((s_bra, s_ket), (i_bra, i_ket))
            },
        )
    }};
}
