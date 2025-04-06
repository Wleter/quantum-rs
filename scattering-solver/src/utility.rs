use faer::{
    dyn_stack::{MemBuffer, MemStack}, linalg::lu, perm::PermRef, unzip, zip, Conj, MatMut, MatRef, Par
};
use serde::Serialize;
use std::{
    fs::{File, create_dir_all},
    io::Write,
    path::Path,
};

#[derive(Clone, Copy, Debug)]
pub struct AngMomentum(pub u32);

pub fn save_data(filename: &str, header: &str, data: &[Vec<f64>]) -> Result<(), std::io::Error> {
    let n = data.first().unwrap().len();
    for values in data {
        assert!(values.len() == n, "Same length data allowed only")
    }

    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("dat");
    let filepath = path.parent().unwrap();

    let mut buf = header.to_string();
    for i in 0..n {
        let line = data
            .iter()
            .fold(String::new(), |s, val| s + &format!("\t{:e}", val[i]));

        buf.push_str(&format!("\n{}", line.trim()));
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

pub fn save_serialize(filename: &str, data: &impl Serialize) -> Result<(), std::io::Error> {
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("json");
    let filepath = path.parent().unwrap();

    let buf = serde_json::to_string(data).unwrap();

    if !Path::new(filepath).exists() {
        create_dir_all(filepath)?;
        println!("created path {}", filepath.display());
    }

    let mut file = File::create(&path)?;
    file.write_all(buf.as_bytes())?;

    println!("saved data on {}", path.display());
    Ok(())
}

pub fn inverse_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
) {
    zip!(out.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    let dim: usize = mat.nrows();

    let mut mem = MemBuffer::new(
        lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(dim, dim, Par::Seq, Default::default())
            .or(lu::partial_pivoting::inverse::inverse_scratch(dim, Par::Seq))
    );
    let stack = MemStack::new(&mut mem);

    lu::partial_pivoting::factor::lu_in_place(
        out.as_mut(),
        perm,
        perm_inv,
        Par::Seq,
        stack,
        Default::default(),
    );

    let perm_ref = unsafe { PermRef::new_unchecked(perm, perm_inv, dim) };

    lu::partial_pivoting::inverse::inverse(
        out.as_mut(),
        perm_ref,
        Par::Seq,
        stack
    );
}

pub fn inverse_inplace_det(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
) -> f64 {
    zip!(out.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    let dim: usize = mat.nrows();

    let (lu_info, _) = lu::partial_pivoting::compute::lu_in_place(
        out.as_mut(),
        perm,
        perm_inv,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            lu_in_place_req::<usize, f64>(dim, dim, Par::Seq, Default::default()).unwrap(),
        )),
        Default::default(),
    );

    let mut determinant = 1.0;
    for diag in out.as_ref().diagonal().column_vector().iter() {
        determinant *= diag
    }

    let perm_ref = unsafe { PermRef::new_unchecked(perm, perm_inv, dim) };

    lu::partial_pivoting::inverse::invert_in_place(
        out.as_mut(),
        perm_ref,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            invert_req::<usize, f64>(dim, dim, Par::Seq).unwrap(),
        )),
    );

    if lu_info.transposition_count & 1 == 0 {
        determinant
    } else {
        -determinant
    }
}

pub fn inverse_inplace_nodes(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
) -> u64 {
    zip!(out.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    let dim: usize = mat.nrows();

    lu::partial_pivoting::compute::lu_in_place(
        out.as_mut(),
        perm,
        perm_inv,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            lu_in_place_req::<usize, f64>(dim, dim, Par::Seq, Default::default()).unwrap(),
        )),
        Default::default(),
    );

    let mut nodes = 0;
    for diag in out.as_ref().diagonal().column_vector().iter() {
        if *diag < 0.0 {
            nodes += 1
        }
    }

    let perm_ref = unsafe { PermRef::new_unchecked(perm, perm_inv, dim) };

    lu::partial_pivoting::inverse::invert_in_place(
        out.as_mut(),
        perm_ref,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            invert_req::<usize, f64>(dim, dim, Par::Seq).unwrap(),
        )),
    );

    nodes
}

/// Should be faster for symmetric matrices but did not observed that.
pub fn inverse_symmetric_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
) {
    let dim: usize = mat.nrows();

    let mut buffer = GlobalPodBuffer::new(temp_mat_req::<f64>(dim, 1).unwrap());
    let col_stack = PodStack::new(&mut buffer);
    let mut sub_diag = temp_mat_uninit::<f64>(dim, 1, col_stack).0;

    let mut buffer = GlobalPodBuffer::new(temp_mat_req::<f64>(dim, dim).unwrap());
    let mat_stack = PodStack::new(&mut buffer);
    let mut mat_temp = temp_mat_uninit::<f64>(dim, dim, mat_stack).0;

    zip!(mat_temp.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    cholesky_in_place(
        mat_temp.as_mut(),
        sub_diag.as_mut().col_mut(0),
        Default::default(),
        perm,
        perm_inv,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            cholesky_in_place_req::<usize, f64>(dim, Par::Seq, Default::default())
                .unwrap(),
        )),
        Default::default(),
    );

    out.fill_zero();
    out.as_mut()
        .diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .for_each(|x| *x = 1.);

    faer::linalg::cholesky::bunch_kaufman::solve::solve_in_place_with_conj(
        mat_temp.as_ref(),
        sub_diag.col(0),
        Conj::No,
        unsafe { PermRef::new_unchecked(perm, perm_inv, dim) },
        out,
        Par::Seq,
        PodStack::new(&mut GlobalPodBuffer::new(
            faer::linalg::cholesky::bunch_kaufman::solve::solve_in_place_req::<usize, f64>(
                dim,
                dim,
                Par::Seq,
            )
            .unwrap(),
        )),
    );
}
