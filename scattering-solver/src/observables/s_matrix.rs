use std::f64::consts::PI;

use faer::{Mat, MatRef, prelude::c64};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct SingleSMatrix {
    s_matrix: Complex64,
    momentum: f64,
}

impl SingleSMatrix {
    pub fn new(s_matrix: Complex64, momentum: f64) -> Self {
        Self { s_matrix, momentum }
    }

    pub fn get_scattering_length(&self) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - self.s_matrix) / (1.0 + self.s_matrix)
    }

    pub fn get_elastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix).norm_sqr()
    }

    pub fn get_inelastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix.norm()).powi(2)
    }

    pub fn observables(&self) -> ScatteringObservables {
        let inelastic_cross_sections = vec![self.get_inelastic_cross_sect()];

        ScatteringObservables {
            entrance: 0,
            scattering_length: self.get_scattering_length(),
            elastic_cross_section: self.get_elastic_cross_sect(),
            inelastic_cross_sections,
        }
    }
}

pub struct SMatrix {
    s_matrix: Mat<c64>,
    momentum: f64,
    entrance: usize,
}

impl SMatrix {
    pub fn new(s_matrix: Mat<c64>, momentum: f64, entrance: usize) -> Self {
        Self {
            s_matrix,
            momentum,
            entrance,
        }
    }

    pub fn new_single(s_matrix: Complex64, momentum: f64) -> Self {
        let mut s_mat = Mat::zeros(1, 1);
        s_mat.as_mut().fill(s_matrix.into());

        Self {
            s_matrix: s_mat,
            momentum,
            entrance: 0,
        }
    }

    pub fn s_matrix(&self) -> MatRef<c64> {
        self.s_matrix.as_ref()
    }

    pub fn get_scattering_length(&self) -> Complex64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - s_element) / (1.0 + s_element)
    }

    pub fn get_elastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        PI / self.momentum.powi(2) * (1.0 - s_element).norm_sqr()
    }

    pub fn get_inelastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)].into();

        PI / self.momentum.powi(2) * (1.0 - s_element.norm()).powi(2)
    }

    pub fn get_inelastic_cross_sect_to(&self, channel: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, channel)].into();

        PI / self.momentum.powi(2) * s_element.norm_sqr()
    }

    pub fn observables(&self) -> ScatteringObservables {
        let mut inelastic_cross_sections = self
            .s_matrix
            .row(self.entrance)
            .iter()
            .map(|s| PI / self.momentum.powi(2) * s.norm_sqr())
            .collect::<Vec<f64>>();
        inelastic_cross_sections[self.entrance] = self.get_inelastic_cross_sect();

        ScatteringObservables {
            entrance: self.entrance,
            scattering_length: self.get_scattering_length(),
            elastic_cross_section: self.get_elastic_cross_sect(),
            inelastic_cross_sections,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScatteringObservables {
    pub entrance: usize,
    pub scattering_length: Complex64,
    pub elastic_cross_section: f64,
    pub inelastic_cross_sections: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScatteringDependence<T> {
    pub parameters: Vec<T>,
    pub observables: Vec<ScatteringObservables>,
}
