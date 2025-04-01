use faer::{MatMut, MatRef};

use super::{LogDerivative, LogDerivativeReference};

pub type DiabaticLogDerivative<'a, S> = LogDerivative<'a, S, Diabatic>;

pub struct Diabatic;

impl LogDerivativeReference for Diabatic {
    fn w_ref(w_c: MatRef<f64>, mut w_ref: MatMut<f64>) {
        w_ref.fill_zero();
        
        w_ref.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_c
                .diagonal()
                .column_vector()
                .iter()
            )
            .for_each(|(w_ref, &w_c)| {
                *w_ref = w_c
            });
    }

    fn imbedding1(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_ref.diagonal()
                .column_vector()
                .iter()
            )
            .for_each(|(y1, &p)| {
                if p < 0.0 {
                    *y1 = -p * 1.0 / f64::tanh(-p * h)
                } else {
                    *y1 = p * 1.0 / f64::tan(p * h)
                }
            });
    }

    fn imbedding2(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_ref.diagonal()
                .column_vector()
                .iter()
            )
            .for_each(|(y1, &p)| {
                if p < 0.0 {
                    *y1 = -p * 1.0 / f64::sinh(-p * h)
                } else {
                    *y1 = p * 1.0 / f64::sin(p * h)
                }
            });
    }

    #[inline]
    fn imbedding3(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        Self::imbedding1(h, w_ref, out);
    }
}
