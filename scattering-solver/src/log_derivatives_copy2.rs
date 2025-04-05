struct LogDerivativeStep<R: LogDerivativeReference> {
    buffer1: Mat<f64>,
    buffer2: Mat<f64>,
    buffer3: Mat<f64>,

    z_matrix: Mat<f64>,
    w_ref: Mat<f64>,

    perm: Vec<usize>,
    perm_inv: Vec<usize>,

    reference: PhantomData<R>
}

impl<R: LogDerivativeReference> LogDerivativeStep<R> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer1: Mat::zeros(size, size),
            buffer2: Mat::zeros(size, size),
            buffer3: Mat::zeros(size, size),
            z_matrix: Mat::zeros(size, size),
            w_ref: Mat::zeros(size, size),
            perm: vec![0; size],
            perm_inv: vec![0; size],
            reference: PhantomData,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        let h = sol.dr / 2.0;

        eq.w_matrix(sol.r + h, &mut self.buffer1);
        R::w_ref(self.buffer1.as_ref(), self.w_ref.as_mut());

        zipped!(self.buffer1.as_mut(), eq.unit.as_ref(), self.w_ref.as_ref())
        .for_each(|unzipped!(b, u, w_ref)| {
            *b = u - h * h / 6. * (w_ref - *b)  // sign change because of different convention
        });

        inverse_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut self.perm, &mut self.perm_inv);

        zipped!(self.buffer2.as_mut(), eq.unit.as_ref())
        .for_each(|unzipped!(b, u)| {
            *b = 6. / (h * h) * (*b - u)
        });
        // buffer2 is a W_tilde(c)

        R::imbedding4(h, self.w_ref.as_ref(), self.buffer1.as_mut());

        zipped!(self.buffer1.as_mut(), self.buffer2.as_ref())
        .for_each(|unzipped!(y4, w_tilde)| {
            *y4 = *y4 + 2. * h / 3. * w_tilde
        });
        // buffer1 is a y_4(a, c)

        R::imbedding1(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zipped!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzipped!(y4, w_tilde)| {
            *y4 = *y4 + 2. * h / 3. * w_tilde
        });
        // buffer3 is a y_1(c, b)

        zipped!(self.buffer1.as_mut(), self.buffer3.as_ref())
        .for_each(|unzipped!(y4, y1)| {
            *y4 = *y4 + y1
        });
        inverse_inplace(self.buffer1.as_ref(), self.z_matrix.as_mut(), &mut self.perm, &mut self.perm_inv);
        // z_matrix is a z(a, b, c)

        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), self.buffer1.as_ref(), self.z_matrix.as_ref(), None, 1.0, faer::Parallelism::None);
        R::imbedding3(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer2.as_mut(), self.buffer3.as_ref(), self.buffer1.as_ref(), None, 1.0, faer::Parallelism::None);
        // buffer2 is a second term in y_1(a, b)
        
        eq.w_matrix(sol.r, &mut self.buffer1);
        R::imbedding1(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zipped!(self.buffer3.as_mut(), self.buffer1.as_ref(), self.w_ref.as_ref())
        .for_each(|unzipped!(y1, w_a, w_ref)| {
            *y1 = *y1 + h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_1(a, c)

        zipped!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzipped!(y1, b)| {
            *y1 = *y1 - b
        });
        // buffer3 is a y_1(a, b)
        
        zipped!(self.buffer3.as_mut(), sol.sol.0.as_ref())
        .for_each(|unzipped!(y1, sol)| {
            *y1 = sol + *y1
        });
        
        let mut nodes = inverse_inplace_nodes(self.buffer3.as_ref(), sol.sol.0.as_mut(), &mut self.perm, &mut self.perm_inv);
        // sol is now (y + y1(a, b))^-1

        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), self.buffer1.as_ref(), self.z_matrix.as_ref(), None, 1.0, faer::Parallelism::None);
        matmul(self.buffer2.as_mut(), self.buffer3.as_ref(), self.buffer1.as_ref(), None, 1.0, faer::Parallelism::None);

        matmul(self.buffer1.as_mut(), sol.sol.0.as_ref(), self.buffer2.as_ref(), None, 1.0, faer::Parallelism::None);
        // buffer1 is now (y + y1(a, b))^-1 * y_2(a, b)

        R::imbedding3(h, self.w_ref.as_ref(), self.buffer2.as_mut());
        matmul(sol.sol.0.as_mut(), self.buffer2.as_ref(), self.z_matrix.as_ref(), None, 1.0, faer::Parallelism::None);
        matmul(self.buffer3.as_mut(), sol.sol.0.as_ref(), self.buffer2.as_ref(), None, 1.0, faer::Parallelism::None);
        
        matmul(sol.sol.0.as_mut(), self.buffer3.as_ref(), self.buffer1.as_ref(), None, 1.0, faer::Parallelism::None);
        // sol is now y_3(a, b) * (y + y1(a, b))^-1 * y_2(a, b)

        R::imbedding3(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), self.buffer1.as_ref(), self.z_matrix.as_ref(), None, 1.0, faer::Parallelism::None);
        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer2.as_mut(), self.buffer3.as_ref(), self.buffer1.as_ref(), None, 1.0, faer::Parallelism::None);
        // buffer2 is a second term in y_4(a, b)
        
        eq.w_matrix(sol.r + sol.dr, &mut self.buffer1);
        R::imbedding4(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zipped!(self.buffer3.as_mut(), self.buffer1.as_ref(), self.w_ref.as_ref())
        .for_each(|unzipped!(y4, w_a, w_ref)| {
            *y4 = *y4 + h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_4(c, b)

        zipped!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzipped!(y4, b)| {
            *y4 = *y4 - b
        });
        // buffer3 is a y_4(a, b)

        zipped!(sol.sol.0.as_mut(), self.buffer3.as_ref())
        .for_each(|unzipped!(y, y4)| {
            *y = y4 - *y
        });
        // sol is y(b)

        if sol.dr < 0. {
            nodes = self.perm.len() as u64 - nodes
        }
        sol.nodes += nodes;

        sol.r += sol.dr;
    }
}