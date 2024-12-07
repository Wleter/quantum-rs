use std::{fs::{create_dir_all, File}, io::Write, path::Path};

#[derive(Clone, Copy, Debug)]
pub struct AngularSpin(pub usize);

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

pub mod faer {
    use faer::{dyn_stack::{GlobalPodBuffer, PodStack}, linalg::lu::{self, partial_pivoting::{compute::lu_in_place_req, inverse::invert_req}}, perm::PermRef, unzipped, zipped, MatMut, MatRef, Parallelism};

    pub fn inverse_inplace(mat: MatRef<f64>, mut out: MatMut<f64>, perm: &mut [usize], perm_inv: &mut [usize]) {
        zipped!(out.as_mut(), mat)
            .for_each(|unzipped!(mut o, m)| o.write(m.read()));

        let dim: usize = mat.nrows();

        lu::partial_pivoting::compute::lu_in_place(
            out.as_mut(), 
            perm, 
            perm_inv, 
            faer::Parallelism::None, 
            PodStack::new(&mut GlobalPodBuffer::new(
                lu_in_place_req::<usize, f64>(
                    dim, 
                    dim,
                    Parallelism::None,
                    Default::default(),
                ).unwrap()
            )),
            Default::default()
        );

        let perm_ref = unsafe {
            PermRef::new_unchecked(perm, perm_inv)
        };

        lu::partial_pivoting::inverse::invert_in_place(
            out.as_mut(), 
            perm_ref, 
            faer::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                invert_req::<usize, f64>(
                    dim, 
                    dim,
                    Parallelism::None,
                ).unwrap()
            ))
        );
    }
}