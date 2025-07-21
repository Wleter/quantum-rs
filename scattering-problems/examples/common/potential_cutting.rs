use std::cell::RefCell;

use faer::{dyn_stack::{MemBuffer, MemStack}, get_global_parallelism, linalg::{self, matmul, temp_mat_scratch, temp_mat_uninit, temp_mat_zeroed}, mat, Mat};
use scattering_solver::potentials::potential::{MatPotential, Potential};


pub struct PotentialCutting<P: MatPotential> {
    cut_after: usize,
    potential: P,
    buffer: RefCell<MemBuffer>,
    full_pot_buffer: RefCell<Mat<f64>>
}

impl<P: MatPotential> PotentialCutting<P> {
    pub fn new(potential: P, cut: usize) -> Self {
        let size = potential.size();
		let par = get_global_parallelism();
        
        let buffer = MemBuffer::new(
            temp_mat_scratch::<f64>(size, 1)
            .and(temp_mat_scratch::<f64>(size, size))
            .and(temp_mat_scratch::<f64>(cut, cut))
            .and(temp_mat_scratch::<f64>(cut, cut))
            .and(
                linalg::evd::self_adjoint_evd_scratch::<f64>(
                    size,
                    linalg::evd::ComputeEigenvectors::Yes,
                    par,
                    Default::default()
                )
            ),
        );

        Self {
            cut_after: cut,
            potential,
            buffer: RefCell::new(buffer),
            full_pot_buffer: RefCell::new(Mat::zeros(size, size)),
        }
    }
}

impl<P: MatPotential> Potential for PotentialCutting<P> {
    type Space = Mat<f64>;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        let mut buffer = self.buffer.borrow_mut();
        let stack = MemStack::new(&mut buffer);
        let size = self.potential.size();
		let par = get_global_parallelism();

        let (mut s, stack) = unsafe { temp_mat_uninit::<f64, _, _>(size, 1, stack) };
        let mut s = mat::AsMatMut::as_mat_mut(&mut s)
            .col_mut(0)
            .as_diagonal_mut();

        let (mut u, stack) = unsafe { temp_mat_uninit::<f64, _, _>(size, size, stack) };
        let mut u = mat::AsMatMut::as_mat_mut(&mut u);

        let (mut sub_s, stack) = temp_mat_zeroed::<f64, _, _>(self.cut_after, self.cut_after, stack);
        let mut sub_s = mat::AsMatMut::as_mat_mut(&mut sub_s);

        let (mut temp, stack) = unsafe { temp_mat_uninit::<f64, _, _>(self.cut_after, self.cut_after, stack) };
        let mut temp = mat::AsMatMut::as_mat_mut(&mut temp);

        self.potential.value_inplace(r, &mut self.full_pot_buffer.borrow_mut());
        let full_pot_buffer = self.full_pot_buffer.borrow();
        
        linalg::evd::self_adjoint_evd(
			full_pot_buffer.as_ref(),
			s.as_mut(),
			Some(u.as_mut()),
			par,
			stack,
			Default::default(),
		).expect("error while doing eigendecomposition");

        let sub_u = u.submatrix(0, 0, self.cut_after, self.cut_after);
        for i in 0..self.cut_after {
            sub_s[(i, i)] = s[i]
        }

        let sub_u = sub_u.qr().compute_Q();

        // println!("{sub_s:?}\n {sub_u:?}\n {:?}", sub_u.transpose());

        matmul::matmul(
            temp.as_mut(), 
            faer::Accum::Replace, 
            sub_s.as_ref(), 
            sub_u.transpose(), 
            1., 
            par
        );

        matmul::matmul(
            value.as_mut(), 
            faer::Accum::Replace, 
            sub_u, 
            temp.as_ref(), 
            1., 
            par
        );
    }

    fn size(&self) -> usize {
        self.cut_after
    }
}