pub mod diabatic;
pub mod johnson;

use std::marker::PhantomData;

use faer::{
    dyn_stack::MemBuffer, linalg::{matmul::matmul, solvers::DenseSolveCore}, prelude::c64, unzip, zip, Accum, ColRef, Mat, MatMut, MatRef, Par
};
use quantum::utility::{
    ratio_riccati_i_deriv, ratio_riccati_k_deriv, riccati_j_deriv, riccati_n_deriv,
};

use crate::{
    boundary::{Boundary, Direction},
    numerovs::{propagator_watcher::PropagatorWatcher, StepRule},
    observables::{bound_states::WaveFunction, s_matrix::SMatrix},
    propagator::{Equation, Propagator, Repr, Solution},
    utility::{get_ldlt_inverse_buffer, inverse_ldlt_inplace, inverse_ldlt_inplace_nodes},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct LogDeriv<T>(pub T);
impl<T> Repr<T> for LogDeriv<T> {}

// doi: 10.1063/1.451472
pub trait LogDerivativeReference {
    fn w_ref(w_c: MatRef<f64>, w_ref: MatMut<f64>);

    fn imbedding1(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>);
    fn imbedding2(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>);
    fn imbedding3(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>);
    fn imbedding4(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>);
}

pub struct LogDerivative<'a, S, R>
where
    S: StepRule<Mat<f64>>,
    R: LogDerivativeReference,
{
    pub(super) equation: Equation<'a, Mat<f64>>,
    pub(super) solution: Solution<LogDeriv<Mat<f64>>>,

    step: LogDerivativeStep<R>,
    step_rule: S,
}

impl<'a, S, R> LogDerivative<'a, S, R>
where
    S: StepRule<Mat<f64>>,
    R: LogDerivativeReference,
{
    pub fn new(mut eq: Equation<'a, Mat<f64>>, boundary: Boundary<Mat<f64>>, step_rule: S) -> Self {
        let r = boundary.r_start;

        eq.buffer_w_matrix(r);
        let dr = match boundary.direction {
            Direction::Inwards => -step_rule.get_step(eq.buffered_w_matrix()),
            Direction::Outwards => step_rule.get_step(eq.buffered_w_matrix()),
            Direction::Step(dr) => dr,
        };

        let sol = Solution {
            r,
            dr,
            sol: LogDeriv(boundary.start_value),
            nodes: 0,
        };

        Self {
            step: LogDerivativeStep::new(eq.potential.size()),
            equation: eq,
            solution: sol,
            step_rule,
        }
    }

    pub fn with_storage(&mut self, storage: WaveLogDerivStorage) {
        self.step.wave_storage = Some(storage)
    }

    pub fn storage(&mut self) -> &Option<WaveLogDerivStorage> {
        &self.step.wave_storage
    }


    pub fn s_matrix(&self) -> SMatrix {
        self.solution.s_matrix(&self.equation)
    }

    pub fn solution(&self) -> &Solution<LogDeriv<Mat<f64>>> {
        &self.solution
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        self.equation.buffer_w_matrix(self.solution.r);

        let dr_new = self.step_rule.get_step(self.equation.buffered_w_matrix());
        self.solution.dr =
            dr_new.clamp(0., 2. * self.solution.dr.abs()) * self.solution.dr.signum();

        if let Some(r) = r {
            if (self.solution.r - r).abs() < self.solution.dr.abs() {
                self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
            }
        }

        self.step.perform_step(&mut self.solution, &self.equation);
    }
}

impl<S, R> Propagator<Mat<f64>, LogDeriv<Mat<f64>>> for LogDerivative<'_, S, R>
where
    S: StepRule<Mat<f64>>,
    R: LogDerivativeReference,
{
    // todo! get minimal step from w_matrix from r and r+dr, r+dr buffered in eq,
    // r buffered in LogDerivativeStep
    fn step(&mut self) -> &Solution<LogDeriv<Mat<f64>>> {
        self.step_r_target(None);

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<Mat<f64>>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        &self.solution
    }

    fn propagate_to_with(
        &mut self,
        r: f64,
        modifier: &mut impl PropagatorWatcher<Mat<f64>, LogDeriv<Mat<f64>>>,
    ) -> &Solution<LogDeriv<Mat<f64>>> {
        modifier.before(&self.solution, &self.equation, r);

        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));

            modifier.after_step(&self.solution, &self.equation);
        }

        modifier.after_prop(&self.solution, &self.equation);

        &self.solution
    }
}

/// https://doi.org/10.1016/0010-4655(94)90200-3
struct LogDerivativeStep<R: LogDerivativeReference> {
    buffer1: Mat<f64>,
    buffer2: Mat<f64>,
    buffer3: Mat<f64>,
    inverse_buffer: MemBuffer,

    z_matrix: Mat<f64>,
    w_ref: Mat<f64>,

    reference: PhantomData<R>,

    wave_storage: Option<WaveLogDerivStorage>
}

impl<R: LogDerivativeReference> LogDerivativeStep<R> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer1: Mat::zeros(size, size),
            buffer2: Mat::zeros(size, size),
            buffer3: Mat::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),

            z_matrix: Mat::zeros(size, size),
            w_ref: Mat::zeros(size, size),

            reference: PhantomData,
            wave_storage: None,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Mat<f64>>>, eq: &Equation<Mat<f64>>) {
        let h = sol.dr / 2.0;

        eq.w_matrix(sol.r + h, &mut self.buffer1);
        R::w_ref(self.buffer1.as_ref(), self.w_ref.as_mut());

        zip!(self.buffer1.as_mut(), eq.unit.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(b, u, w_ref)| {
            *b = u - h * h / 6. * (w_ref - *b)  // sign change because of different convention
        });

        inverse_ldlt_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        zip!(self.buffer2.as_mut(), eq.unit.as_ref())
        .for_each(|unzip!(b, u)| {
            *b = 6. / (h * h) * (*b - u)
        });
        // buffer2 is a W_tilde(c)

        R::imbedding4(h, self.w_ref.as_ref(), self.buffer1.as_mut());

        zip!(self.buffer1.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, w_tilde)| {
            *y4 = *y4 + 2. * h / 3. * w_tilde
        });
        // buffer1 is a y_4(a, c)

        R::imbedding1(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, w_tilde)| {
            *y4 = *y4 + 2. * h / 3. * w_tilde
        });
        // buffer3 is a y_1(c, b)

        zip!(self.buffer1.as_mut(), self.buffer3.as_ref())
        .for_each(|unzip!(y4, y1)| {
            *y4 = *y4 + y1
        });
        inverse_ldlt_inplace(self.buffer1.as_ref(), self.z_matrix.as_mut(), &mut self.inverse_buffer);
        // z_matrix is a z(a, b, c)

        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), Accum::Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Par::Seq);
        R::imbedding3(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer2.as_mut(), Accum::Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Par::Seq);
        // buffer2 is a second term in y_1(a, b)
        
        eq.w_matrix(sol.r, &mut self.buffer1);
        R::imbedding1(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zip!(self.buffer3.as_mut(), self.buffer1.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(y1, w_a, w_ref)| {
            *y1 = *y1 + h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_1(a, c)

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y1, b)| {
            *y1 = *y1 - b
        });
        // buffer3 is a y_1(a, b)
        
        zip!(self.buffer3.as_mut(), sol.sol.0.as_ref())
        .for_each(|unzip!(y1, sol)| {
            *y1 = sol + *y1
        });
        
        let mut nodes = inverse_ldlt_inplace_nodes(self.buffer3.as_ref(), sol.sol.0.as_mut(), &mut self.inverse_buffer);
        // sol is now (y + y1(a, b))^-1

        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), Accum::Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Par::Seq);
        matmul(self.buffer2.as_mut(), Accum::Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Par::Seq);

        matmul(self.buffer1.as_mut(), Accum::Replace, sol.sol.0.as_ref(), self.buffer2.as_ref(), 1.0, Par::Seq);
        // buffer1 is now (y + y1(a, b))^-1 * y_2(a, b)

        if let Some(storage) = &mut self.wave_storage {
            storage.push(sol.r, self.buffer1.as_ref());
        }

        R::imbedding3(h, self.w_ref.as_ref(), self.buffer2.as_mut());
        matmul(sol.sol.0.as_mut(), Accum::Replace, self.buffer2.as_ref(), self.z_matrix.as_ref(), 1.0, Par::Seq);
        matmul(self.buffer3.as_mut(), Accum::Replace, sol.sol.0.as_ref(), self.buffer2.as_ref(), 1.0, Par::Seq);
        
        matmul(sol.sol.0.as_mut(), Accum::Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Par::Seq);
        // sol is now y_3(a, b) * (y + y1(a, b))^-1 * y_2(a, b)

        R::imbedding3(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer3.as_mut(), Accum::Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Par::Seq);
        R::imbedding2(h, self.w_ref.as_ref(), self.buffer1.as_mut());
        matmul(self.buffer2.as_mut(), Accum::Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Par::Seq);
        // buffer2 is a second term in y_4(a, b)
        
        eq.w_matrix(sol.r + sol.dr, &mut self.buffer1);
        R::imbedding4(h, self.w_ref.as_ref(), self.buffer3.as_mut());

        zip!(self.buffer3.as_mut(), self.buffer1.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(y4, w_a, w_ref)| {
            *y4 = *y4 + h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_4(c, b)

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, b)| {
            *y4 = *y4 - b
        });
        // buffer3 is a y_4(a, b)

        zip!(sol.sol.0.as_mut(), self.buffer3.as_ref())
        .for_each(|unzip!(y, y4)| {
            *y = y4 - *y
        });
        // sol is y(b)

        let dim = eq.potential.size();
        if sol.dr < 0. {
            nodes = dim as u64 - nodes
        }
        sol.nodes += nodes;

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

                j_deriv_last[(i, i)] = ratio_i * momentum;
                j_last[(i, i)] = 1.0;
                n_deriv_last[(i, i)] = ratio_k * momentum;
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

pub struct WaveLogDerivStorage {
    distances: Vec<f64>,
    connections: Vec<Mat<f64>>,
    last_first: bool,
}

impl WaveLogDerivStorage {
    pub fn new(last_first: bool) -> Self {
        Self {
            distances: Vec::new(),
            connections: Vec::new(),
            last_first,
        }
    }

    pub fn push(&mut self, r: f64, connection: MatRef<f64>) {
        let byte_size = self.connections.len() * 8 * connection.nrows() * connection.ncols();

        if byte_size > (8 * 1024 * 1024 * 1024) {
            panic!("Size overload todo! disk saving")
        }

        self.distances.push(r);
        self.connections.push(connection.cloned());
    }

    pub fn reconstruct(&self, wave_init: ColRef<f64>) -> WaveFunction {
        let mut values = Vec::with_capacity(self.connections.len());
        let mut wave_recent = wave_init.cloned();

        let distances = if self.last_first {
            for c in self.connections.iter().rev() {
                wave_recent = c * &wave_recent;
                values.push(wave_recent.iter().copied().collect())
            }

            self.distances.iter().rev().copied().collect()
        } else {
            for c in self.connections.iter() {
                wave_recent = c * &wave_recent;
                values.push(wave_recent.iter().copied().collect())
            }

            self.distances.clone()
        };
        
        WaveFunction { distances, values }
    }
}