use std::{fs::{create_dir_all, File}, io::Write, path::Path};
use serde::Serialize;

use faer::{dyn_stack::{MemBuffer, MemStack}, linalg::{self, cholesky, lu, temp_mat_scratch, temp_mat_uninit, temp_mat_zeroed}, perm::PermRef, prelude::*, unzip, zip};

#[derive(Clone, Copy, Debug)]
pub struct AngMomentum(pub u32);

pub fn save_data(
    filename: &str,
    header: &str,
    data: &[Vec<f64>],
) -> Result<(), std::io::Error> {

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
        let line = data.iter()
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

pub fn save_serialize(
    filename: &str,
    data: &impl Serialize,
) -> Result<(), std::io::Error> {
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
    assert!(mat.nrows() == mat.ncols());
    let dim: usize = mat.nrows();
    zip!(out.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    lu::partial_pivoting::factor::lu_in_place(
        out.as_mut(),
        perm,
        perm_inv,
        faer::Par::Seq,
        MemStack::new(&mut MemBuffer::new(
            lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
                dim,
                dim,
                faer::Par::Seq,
                Default::default(),
            ),
        )),
        default(),
    );

    let mut buffer = MemBuffer::new(temp_mat_scratch::<f64>(dim, dim)
        .and(temp_mat_scratch::<f64>(dim, dim)));
    let mut stack = MemStack::new(&mut buffer);

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, &mut stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    let (mut u, _) = temp_mat_zeroed::<f64, _, _>(dim, dim, stack);
    let mut u = mat::AsMatMut::as_mat_mut(&mut u);

    u.copy_from_triangular_upper(&out);

    zip!(&mut l, &out).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(l, o)| *l = *o);
    zip!(&mut l).for_each_triangular_upper(linalg::zip::Diag::Skip, |unzip!(x)| *x = 0.);
    l.as_mut().diagonal_mut().fill(1.);

    let perm_ref = unsafe { PermRef::new_unchecked(perm, perm_inv, dim) };

    lu::partial_pivoting::inverse::inverse(
        out.as_mut(),
        l.as_ref(),
        u.as_ref(),
        perm_ref,
        faer::Par::Seq,
        MemStack::new(&mut MemBuffer::new(
            lu::partial_pivoting::inverse::inverse_scratch::<usize, f64>(dim, faer::Par::Seq),
        )),
    );
}

/// Should be faster for symmetric matrices but did not observed that.
pub fn inverse_symmetric_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
) {
    let dim: usize = mat.nrows();

    let mut buffer = MemBuffer::new(
        temp_mat_scratch::<f64>(dim, 1)
            .and(temp_mat_scratch::<f64>(dim, 1))
            .and(temp_mat_scratch::<f64>(dim, dim)),
    );
    let mut stack = MemStack::new(&mut buffer);
    let (mut sub_diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, &mut stack) };
    let mut sub_diag = mat::AsMatMut::as_mat_mut(&mut sub_diag).col_mut(0).as_diagonal_mut();

    let (mut diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut diag = mat::AsMatMut::as_mat_mut(&mut diag).col_mut(0).as_diagonal_mut();

    let (mut l, _) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    zip!(&mut l, &mat).for_each(|unzip!(l, m)| *l = *m);

    cholesky::bunch_kaufman::factor::cholesky_in_place(
        l.as_mut(),
        sub_diag.as_mut(),
        Default::default(),
        perm,
        perm_inv,
        faer::Par::Seq,
        MemStack::new(&mut MemBuffer::new(
            cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, f64>(
                dim,
                faer::Par::Seq,
                Default::default(),
            ),
        )),
        Default::default(),
    );

    diag.copy_from(l.as_ref().diagonal());
    l.as_mut().diagonal_mut().fill(1.);
    zip!(&mut l).for_each_triangular_upper(linalg::zip::Diag::Skip, |unzip!(x)| *x = 0.);

    let perm_ref = unsafe { PermRef::new_unchecked(perm, perm_inv, dim) };

    cholesky::bunch_kaufman::inverse::inverse(
        out.as_mut(),
        l.as_ref(),
        diag.as_ref(),
        sub_diag.as_ref(),
        perm_ref,
        Par::Seq,
        MemStack::new(&mut MemBuffer::new(
            cholesky::bunch_kaufman::inverse::inverse_scratch::<usize, f64>(dim, faer::Par::Seq),
        )),
    );
}
