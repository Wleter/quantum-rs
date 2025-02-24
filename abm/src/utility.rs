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
