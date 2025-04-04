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

        zipped!(self.buffer1.as_mut(), eq.unit.as_ref())
        .for_each(|unzipped!(b, u)| {
            *b = u + h * h / 6. * *b // different sign since W(c) is -W(c) in our notation
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
        
        sol.nodes += inverse_inplace_nodes(self.buffer3.as_ref(), sol.sol.0.as_mut(), &mut self.perm, &mut self.perm_inv);
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

        sol.r += sol.dr;
    }
}

impl Solution<LogDeriv<Mat<f64>>> {
    pub fn s_matrix(&self, eq: &Equation<Mat<f64>>) -> SMatrix {
        let size = eq.potential.size();
        let r = self.r;
        let log_deriv = self.sol.0.as_ref();

        let asymptotic = &eq.asymptotic(r);

        let is_open_channel = asymptotic
            .iter()
            .map(|&val| val < eq.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .iter()
            .map(|&val| (2.0 * eq.mass * (eq.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = Mat::zeros(size, size);
        let mut j_deriv_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_deriv_last = Mat::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            let l = eq.asymptotic.centrifugal[i].0;
            if is_open_channel[i] {
                let (j_riccati, j_deriv_riccati) = riccati_j_deriv(l, momentum * r);
                let (n_riccati, n_deriv_riccati) = riccati_n_deriv(l, momentum * r);

                j_last[(i, i)] = j_riccati / momentum.sqrt();
                j_deriv_last[(i, i)] = j_deriv_riccati * momentum.sqrt();
                n_last[(i, i)] = n_riccati / momentum.sqrt();
                n_deriv_last[(i, i)] = n_deriv_riccati * momentum.sqrt();
            } else {
                let ratio_i = ratio_riccati_i_deriv(l, momentum * r);
                let ratio_k = ratio_riccati_k_deriv(l, momentum * r);

                j_deriv_last[(i, i)] = ratio_i;
                j_last[(i, i)] = 1.0;
                n_deriv_last[(i, i)] = ratio_k;
                n_last[(i, i)] = 1.0;
            }
        }

        let denominator = (log_deriv * n_last - n_deriv_last).partial_piv_lu();
        let denominator = denominator.inverse();

        let k_matrix = -denominator * (log_deriv * j_last - j_deriv_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while !is_open_channel[i_full] {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while !is_open_channel[j_full] {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = c64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = Mat::<c64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).partial_piv_lu();
        let denominator = denominator.inverse();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == eq.asymptotic.entrance)
            .expect("Closed entrance channel")
            .0;

        SMatrix::new(s_matrix, momenta[eq.asymptotic.entrance], entrance)
    }
}