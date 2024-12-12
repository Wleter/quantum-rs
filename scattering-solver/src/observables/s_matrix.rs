use std::f64::consts::PI;

use faer::{prelude::c64, Mat};
use num::complex::Complex64;

pub trait HasSMatrix {
    fn get_scattering_length(&self) -> Complex64;

    fn get_elastic_cross_sect(&self) -> f64;

    fn get_inelastic_cross_sect(&self) -> f64;

    fn get_inelastic_cross_sect_to(&self, channel: usize) -> f64;
}

#[derive(Clone, Debug)]
pub struct SingleSMatrix {
    s_matrix: Complex64,
    momentum: f64,
}

impl SingleSMatrix {
    pub fn new(s_matrix: Complex64, momentum: f64) -> Self {
        Self { s_matrix, momentum }
    }
}

impl HasSMatrix for SingleSMatrix {
    fn get_scattering_length(&self) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - self.s_matrix) / (1.0 + self.s_matrix)
    }

    fn get_elastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix).norm_sqr()
    }

    fn get_inelastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix.norm()).powi(2)
    }

    fn get_inelastic_cross_sect_to(&self, _channel: usize) -> f64 {
        0.
    }
}

pub struct FaerSMatrix {
    s_matrix: Mat<c64>,
    momenta: Vec<f64>,
    entrance: usize,
}

impl FaerSMatrix {
    pub fn new(s_matrix: Mat<c64>, momenta: Vec<f64>, entrance: usize) -> Self {
        Self {
            s_matrix,
            momenta,
            entrance,
        }
    }
}

impl HasSMatrix for FaerSMatrix {
    fn get_scattering_length(&self) -> num::complex::Complex64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        1.0 / Complex64::new(0.0, self.momenta[self.entrance]) * (1.0 - s_element) / (1.0 + s_element)
    }

    fn get_elastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element).norm_sqr()
    }

    fn get_inelastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element.norm()).powi(2)
    }

    fn get_inelastic_cross_sect_to(&self, channel: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, channel)].into();

        PI / self.momenta[channel].powi(2) * s_element.norm().powi(2)
    }
}
