use std::marker::PhantomData;

use super::potential::{Potential, SubPotential};

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Debug, Clone)]
pub struct MultiCoupling<A, P: Potential> {
    potentials: Vec<(P, usize, usize)>,
    size: usize,
    phantom: PhantomData<A>,
}

impl<A, P: Potential> MultiCoupling<A, P>
{
    /// Creates new multi coupling potential with given vector of potentials with their coupling indices in potential matrix.
    /// If `symmetric` is true, the coupling matrix will be symmetric.
    pub fn new(size: usize, potentials: Vec<(P, usize, usize)>) -> Self {
        for p in &potentials {
            assert!(p.1 < size);
            assert!(p.2 < size);
        }
        
        Self {
            potentials,
            size,
            phantom: PhantomData
        }
    }

    pub fn new_neighboring(couplings: Vec<P>) -> Self {
        let size = couplings.len() + 1;

        let numbered_potentials = couplings
            .into_iter()
            .enumerate()
            .map(|(i, potential)| (potential, i, i + 1))
            .collect();

        Self::new(size, numbered_potentials)
    }
}

use faer::Mat;

impl<P: Potential<Space = f64>> Potential for MultiCoupling<Mat<f64>, P> {
    type Space = Mat<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Mat<f64>) {
        value.fill_zero();
        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut(*i, *j));
        }

        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut(*j, *i));
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<P: SubPotential<Space = f64>> SubPotential for MultiCoupling<Mat<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Mat<f64>) {
        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut(*i, *j));
        }

        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut(*j, *i));
        }
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<P: Potential<Space = f64>> Potential for MultiCoupling<Array2<f64>, P> {
    type Space = Array2<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut Array2<f64>) {
        value.fill(0.);

        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_inplace(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }
    
    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(feature = "ndarray")]
impl<P: SubPotential<Space = f64>> SubPotential for MultiCoupling<Array2<f64>, P> {
    fn value_add(&self, r: f64, value: &mut Array2<f64>) {
        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_add(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "nalgebra")]
impl<P: Potential<Space = f64>> Potential for MultiCoupling<DMatrix<f64>, P> {
    type Space = DMatrix<f64>;
    
    fn value_inplace(&self, r: f64, value: &mut DMatrix<f64>) {
        value.fill(0.);
        
        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_inplace(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(feature = "nalgebra")]
impl<P: SubPotential<Space = f64>> SubPotential for MultiCoupling<DMatrix<f64>, P> {
    fn value_add(&self, r: f64, value: &mut DMatrix<f64>) {
        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_add(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: Potential<Space = f64>> Potential for MultiCoupling<SMatrix<f64, N, N>, P> {
    type Space = SMatrix<f64, N, N>;
    
    fn value_inplace(&self, r: f64, value: &mut SMatrix<f64, N, N>) {
        value.fill(0.);

        for (p, i, j) in &self.potentials {
            p.value_inplace(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_inplace(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, P: SubPotential<Space = f64>> SubPotential for MultiCoupling<SMatrix<f64, N, N>, P> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        for (p, i, j) in &self.potentials {
            p.value_add(r, value.get_mut((*i, *j)).unwrap());
        }

        if self.symmetric {
            for (p, i, j) in &self.potentials {
                p.value_add(r, value.get_mut((*j, *i)).unwrap());
            }
        }
    }
}
