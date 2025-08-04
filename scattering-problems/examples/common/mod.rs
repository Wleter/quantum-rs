use scattering_solver::potentials::{composite_potential::Composite, potential::{ScaledPotential, SimplePotential, SubPotential}};
use std::fmt::Display;

pub mod srf_rb_functionality;

#[allow(unused)]
#[derive(Clone, Debug, Copy)]
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
    pub fn scale<P: SimplePotential + Clone>(
        &self,
        pes: &[(u32, P)],
        scaling: f64,
    ) -> Vec<(u32, ScaledPotential<P>)> {
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
#[derive(Debug, Clone, Default)]
pub struct Scalings {
    pub scaling_types: Vec<ScalingType>,
    pub scalings: Vec<f64>,
}

#[allow(unused)]
impl Scalings {
    pub fn scale<P: SimplePotential + Clone>(
        &self,
        pes: &[(u32, P)],
    ) -> Vec<(u32, ScaledPotential<P>)> {
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

pub struct Morphing {
    pub lambdas: Vec<u32>,
    pub scalings: Vec<f64>,
}

impl Morphing {
    pub fn morph<P: SimplePotential + Clone + SubPotential>(
        &self,
        pes: &[(u32, P)],
    ) -> Vec<(u32, Composite<ScaledPotential<P>>)> {
        let lambda_max = self.lambdas.iter().max().unwrap();
        let lambda_pes_max = pes.iter().max_by_key(|x| x.0).unwrap();

        (0..(lambda_max + lambda_pes_max.0)).into_iter()
            .map(|lambda| {
                let mut potentials = Vec::new();
                for (lambda_morph, &scaling) in self.lambdas.iter().zip(&self.scalings) {
                    match lambda_morph {
                        0 => {
                            if let Some(p) = pes.iter().find(|x| x.0 == lambda) {
                                potentials.push(ScaledPotential::new(p.1.clone(), scaling))
                            }
                        },
                        1 => {
                            if let Some(p) = pes.iter().find(|x| x.0 == lambda + 1) {
                                let scaling = scaling * (lambda + 1) as f64 / (2 * lambda + 3) as f64; 
                                potentials.push(ScaledPotential::new(p.1.clone(), scaling));
                            }
                            if let Some(p) = pes.iter().find(|x| x.0 + 1 == lambda) {
                                let scaling = scaling * lambda as f64 / (2 * lambda - 1) as f64; 
                                potentials.push(ScaledPotential::new(p.1.clone(), scaling));
                            }
                        },
                        2 => {
                            if let Some(p) = pes.iter().find(|x| x.0 == lambda + 2) {
                                let scaling = 1.5 * scaling 
                                    * ((lambda + 1) * (lambda + 2)) as f64
                                    / ((2 * lambda + 3) * (2 * lambda + 5)) as f64;

                                potentials.push(ScaledPotential::new(p.1.clone(), scaling));
                            }
                            if let Some(p) = pes.iter().find(|x| x.0 == lambda) {
                                let scaling = scaling
                                    * (lambda * (lambda + 1)) as f64
                                    / ((2 * lambda - 1) * (2 * lambda + 1)) as f64;
                                potentials.push(ScaledPotential::new(p.1.clone(), scaling));
                            }
                            if let Some(p) = pes.iter().find(|x| x.0 + 2 == lambda) {
                                let scaling = 1.5 * scaling
                                    * (lambda * (lambda - 1)) as f64
                                    / ((2 * lambda - 3) * (2 * lambda - 1)) as f64;

                                potentials.push(ScaledPotential::new(p.1.clone(), scaling));
                            }
                        },
                        _ => unimplemented!("higher order morphs not supported yet")
                    }
                }

                (lambda, potentials)
            })
            .filter_map(|x| if x.1.is_empty() {
                None
            } else {
                Some((x.0, Composite::from_vec(x.1)))
            })
            .collect()
    }
}



#[allow(unused)]
#[derive(Clone, Copy)]
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
