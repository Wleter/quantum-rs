use scattering_solver::potentials::potential::{ScaledPotential, SimplePotential};
use std::fmt::Display;

pub mod srf_rb_functionality;

#[allow(unused)]
#[derive(Clone)]
pub enum ScalingType {
    Full,
    Isotropic,
    Anisotropic,
    Legendre(u32),
}

impl Display for ScalingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalingType::Full => write!(f, "full"),
            ScalingType::Isotropic => write!(f, "isotropic"),
            ScalingType::Anisotropic => write!(f, "anisotropic"),
            ScalingType::Legendre(lambda) => write!(f, "legendre_{lambda}"),
        }
    }
}

#[allow(unused)]
impl ScalingType {
    pub fn scale(
        &self,
        pes: &[(u32, impl SimplePotential + Clone)],
        scaling: f64,
    ) -> Vec<(u32, impl SimplePotential + Clone)> {
        pes.iter()
            .map(|(lambda, p)| {
                (
                    *lambda,
                    ScaledPotential {
                        potential: p.clone(),
                        scaling: self.scaling(*lambda, scaling),
                    },
                )
            })
            .collect()
    }

    pub fn scaling(&self, lambda: u32, scaling: f64) -> f64 {
        match self {
            ScalingType::Full => scaling,
            ScalingType::Isotropic => {
                if lambda == 0 {
                    scaling
                } else {
                    1.
                }
            }
            ScalingType::Anisotropic => {
                if lambda != 0 {
                    scaling
                } else {
                    1.
                }
            }
            ScalingType::Legendre(l) => {
                if lambda == *l {
                    scaling
                } else {
                    1.
                }
            }
        }
    }
}

#[allow(unused)]
pub struct Scalings {
    pub scalings: Vec<f64>,
    pub scaling_types: Vec<ScalingType>,
}

#[allow(unused)]
impl Scalings {
    pub fn scale(
        &self,
        pes: &[(u32, impl SimplePotential + Clone)],
    ) -> Vec<(u32, impl SimplePotential + Clone)> {
        pes.iter()
            .map(|(lambda, p)| {
                (
                    *lambda,
                    ScaledPotential {
                        potential: p.clone(),
                        scaling: self
                            .scaling_types
                            .iter()
                            .zip(self.scalings.iter())
                            .map(|(t, s)| t.scaling(*lambda, *s))
                            .fold(1., |acc, x| acc * x),
                    },
                )
            })
            .collect()
    }
}

#[allow(unused)]
pub enum PotentialType {
    Singlet,
    Triplet,
}

impl Display for PotentialType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PotentialType::Singlet => write!(f, "singlet"),
            PotentialType::Triplet => write!(f, "triplet"),
        }
    }
}
