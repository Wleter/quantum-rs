use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq}, linalg::{self, cholesky::{self}, lu, temp_mat_scratch, temp_mat_uninit, temp_mat_zeroed}, mat, perm::PermRef, unzip, zip, MatMut, MatRef, Par
};
use serde::Serialize;
use core::slice;
use std::{
    fs::{create_dir_all, File}, io::Write, path::Path
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

pub fn get_ldlt_inverse_buffer(size: usize) -> MemBuffer {
    MemBuffer::new(
        temp_mat_scratch::<f64>(size, 1)
            .and(temp_mat_scratch::<f64>(size, size))
            .and(StackReq::new::<usize>(2 * size))
            .and(
                cholesky::ldlt::factor::cholesky_in_place_scratch::<f64>(
                    size,
                    faer::Par::Seq,
                    Default::default(),
                ).or(
                    cholesky::ldlt::inverse::inverse_scratch::<f64>(size, faer::Par::Seq)
                ),
            )
    )
}

pub fn get_lblt_inverse_buffer(size: usize) -> MemBuffer {
    MemBuffer::new(
        temp_mat_scratch::<f64>(size, size)
            .and(temp_mat_scratch::<f64>(size, 1))
            .and(temp_mat_scratch::<f64>(size, 1))
            .and(StackReq::new::<usize>(2 * size))
            .and(
                cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, f64>(
                    size,
                    faer::Par::Seq,
                    Default::default(),
                ).or(
                    cholesky::bunch_kaufman::inverse::inverse_scratch::<usize, f64>(size, faer::Par::Seq)
                ),
            )
    )
}

pub fn get_lu_inverse_buffer(size: usize) -> MemBuffer {
    MemBuffer::new(
        temp_mat_scratch::<f64>(size, size)
            .and(temp_mat_scratch::<f64>(size, size))
            .and(StackReq::new::<usize>(2 * size))
            .and(
                lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
                size,
                size,
                faer::Par::Seq,
                Default::default(),
            ).or(
                lu::partial_pivoting::inverse::inverse_scratch::<usize, f64>(size, faer::Par::Seq)
            )
        ),
    )
}

pub fn inverse_lu_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    buffer: &mut MemBuffer,
) {
    assert!(mat.nrows() == mat.ncols());
    let dim: usize = mat.nrows();
    zip!(out.as_mut(), mat).for_each(|unzip!(o, m)| *o = *m);

    let mut stack = MemStack::new(buffer);

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, &mut stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    let (mut u, stack) = temp_mat_zeroed::<f64, _, _>(dim, dim, stack);
    let mut u = mat::AsMatMut::as_mat_mut(&mut u);

    let (perm, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm = unsafe { slice::from_raw_parts_mut(perm.as_mut_ptr() as *mut usize, dim) };
    let (perm_inv, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm_inv = unsafe { slice::from_raw_parts_mut(perm_inv.as_mut_ptr() as *mut usize, dim) };

    lu::partial_pivoting::factor::lu_in_place(
        out.as_mut(),
        perm,
        perm_inv,
        faer::Par::Seq,
        stack,
        Default::default(),
    );

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
        stack,
    );
}

/// Should be faster for symmetric matrices but did not observed that.
pub fn inverse_lblt_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    buffer: &mut MemBuffer
) {
    let dim: usize = mat.nrows();
    let stack = MemStack::new(buffer);

    let (mut diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut diag = mat::AsMatMut::as_mat_mut(&mut diag).col_mut(0).as_diagonal_mut();

    let (mut sub_diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut sub_diag = mat::AsMatMut::as_mat_mut(&mut sub_diag).col_mut(0).as_diagonal_mut();

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    let (perm, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm = unsafe { slice::from_raw_parts_mut(perm.as_mut_ptr() as *mut usize, dim) };
    let (perm_inv, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm_inv = unsafe { slice::from_raw_parts_mut(perm_inv.as_mut_ptr() as *mut usize, dim) };

    zip!(&mut l, &mat).for_each(|unzip!(l, m)| *l = *m);
    cholesky::bunch_kaufman::factor::cholesky_in_place(
        l.as_mut(),
        sub_diag.as_mut(),
        Default::default(),
        perm,
        perm_inv,
        faer::Par::Seq,
        stack,
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
        stack,
    );

    for j in 0..dim {
		for i in 0..j {
			out[(i, j)] = out[(j, i)];
		}
	}
}

pub fn inverse_lblt_inplace_nodes(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    buffer: &mut MemBuffer
) -> u64 {
    let dim: usize = mat.nrows();
    let stack = MemStack::new(buffer);

    let (mut diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut diag = mat::AsMatMut::as_mat_mut(&mut diag).col_mut(0).as_diagonal_mut();

    let (mut sub_diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut sub_diag = mat::AsMatMut::as_mat_mut(&mut sub_diag).col_mut(0).as_diagonal_mut();

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    let (perm, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm = unsafe { slice::from_raw_parts_mut(perm.as_mut_ptr() as *mut usize, dim) };
    let (perm_inv, stack) = stack.make_aligned_uninit::<usize>(dim, align_of::<usize>());
    let perm_inv = unsafe { slice::from_raw_parts_mut(perm_inv.as_mut_ptr() as *mut usize, dim) };

    zip!(&mut l, &mat).for_each(|unzip!(l, m)| *l = *m);

    cholesky::bunch_kaufman::factor::cholesky_in_place(
        l.as_mut(),
        sub_diag.as_mut(),
        Default::default(),
        perm,
        perm_inv,
        faer::Par::Seq,
        stack,
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
        stack,
    );

    for j in 0..dim {
		for i in 0..j {
			out[(i, j)] = out[(j, i)];
		}
	}

    let mut negatives = 0;

    let mut block = false;
    let mut a = 0.;

    for (&d, &s) in diag.as_ref().column_vector().iter().zip(sub_diag.as_ref().column_vector().iter()) {
        if s == 0. && d < 0. {
            negatives += 1;
        }
        if s != 0. && !block {
            a = d;
            block = true
        } else if block {
            if a * d > s * s && a + d < 0. {
                negatives += 2
            } 
            if a * d < s * s {
                negatives += 1
            }
        }
    }

    negatives
}

pub fn inverse_ldlt_inplace(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    buffer: &mut MemBuffer
) {
    let dim: usize = mat.nrows();

    let stack = MemStack::new(buffer);

    let (mut diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut diag = mat::AsMatMut::as_mat_mut(&mut diag).col_mut(0).as_diagonal_mut();

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    zip!(&mut l, &mat).for_each(|unzip!(l, m)| *l = *m);

    cholesky::ldlt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        faer::Par::Seq,
        stack,
        Default::default(),
    ).unwrap_or_else(|_| panic!("Could not ldlt decomposition {:?}", mat));

    diag.copy_from(l.as_ref().diagonal());
    l.as_mut().diagonal_mut().fill(1.);
    zip!(&mut l).for_each_triangular_upper(linalg::zip::Diag::Skip, |unzip!(x)| *x = 0.);

    cholesky::ldlt::inverse::inverse(
        out.as_mut(),
        l.as_ref(),
        diag.as_ref(),
        Par::Seq,
        stack,
    );

    for j in 0..dim {
		for i in 0..j {
			out[(i, j)] = out[(j, i)];
		}
	}
}

pub fn inverse_ldlt_inplace_nodes(
    mat: MatRef<f64>,
    mut out: MatMut<f64>,
    buffer: &mut MemBuffer
) -> u64 {
    let dim: usize = mat.nrows();

    let stack = MemStack::new(buffer);

    let (mut diag, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, 1, stack) };
    let mut diag = mat::AsMatMut::as_mat_mut(&mut diag).col_mut(0).as_diagonal_mut();

    let (mut l, stack) = unsafe { temp_mat_uninit::<f64, _, _>(dim, dim, stack) };
    let mut l = mat::AsMatMut::as_mat_mut(&mut l);

    zip!(&mut l, &mat).for_each(|unzip!(l, m)| *l = *m);

    cholesky::ldlt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        faer::Par::Seq,
        stack,
        Default::default(),
    ).unwrap_or_else(|_| panic!("Could not ldlt decomposition {:?}", mat));

    diag.copy_from(l.as_ref().diagonal());
    l.as_mut().diagonal_mut().fill(1.);
    zip!(&mut l).for_each_triangular_upper(linalg::zip::Diag::Skip, |unzip!(x)| *x = 0.);

    let mut negative = 0;
    for &d in diag.as_mut().column_vector_mut().iter() {
        if d < 0. {
            negative += 1;
        }
    }

    cholesky::ldlt::inverse::inverse(
        out.as_mut(),
        l.as_ref(),
        diag.as_ref(),
        Par::Seq,
        stack,
    );

    for j in 0..dim {
		for i in 0..j {
			out[(i, j)] = out[(j, i)];
		}
	}
    
    negative
}

#[cfg(test)]
mod test {
    use faer::{linalg::solvers::DenseSolveCore, Mat};
    use rand::{distr::Uniform, rng, Rng};

    use crate::utility::{get_lblt_inverse_buffer, get_ldlt_inverse_buffer, inverse_lblt_inplace, inverse_ldlt_inplace};

    use super::{get_lu_inverse_buffer, inverse_lu_inplace};

    #[test]
    fn test_inverse_inplace() {
        let mut rng = rng();
        let size = 10;

        let mat = Mat::from_fn(size, size, |_, _| rng.sample(Uniform::new(-10., 10.).unwrap()));
        let mut buffer = get_lu_inverse_buffer(size);
        let mut out = Mat::zeros(size, size);

        inverse_lu_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        assert_eq!(out, mat.partial_piv_lu().inverse());
    }

    #[test]
    fn test_inverse_ldlt_inplace() {
        let mut rng = rng();
        let size = 10;

        let mut mat = Mat::from_fn(size, size, |_, _| rng.sample(Uniform::new(-10., 10.).unwrap()));
        for j in 0..size {
            for i in 0..j {
                mat[(i, j)] = mat[(j, i)];
            }
        }

        let mut buffer = get_ldlt_inverse_buffer(size);
        let mut out = Mat::zeros(size, size);

        inverse_ldlt_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        assert_eq!(out, mat.ldlt(faer::Side::Lower).unwrap().inverse());
    }

    #[test]
    fn test_inverse_lblt_inplace() {
        let mut rng = rng();
        let size = 10;

        let mut mat = Mat::from_fn(size, size, |_, _| rng.sample(Uniform::new(-10., 10.).unwrap()));
        for j in 0..size {
            for i in 0..j {
                mat[(i, j)] = mat[(j, i)];
            }
        }

        let mut buffer = get_lblt_inverse_buffer(size);
        let mut out = Mat::zeros(size, size);

        inverse_lblt_inplace(mat.as_ref(), out.as_mut(), &mut buffer);

        assert_eq!(out, mat.lblt(faer::Side::Lower).inverse());
    }
}