use super::potential::{Potential, SubPotential};

/// Composite potential that has value equal to the sum of its components
#[derive(Debug, Clone)]
pub struct Composite<P: SubPotential> {
    main_potential: P,
    potentials: Vec<P>,
}

impl<P: SubPotential> Composite<P> {
    pub fn new(main_potential: P) -> Self {
        Self {
            main_potential,
            potentials: Vec::new(),
        }
    }

    pub fn add_potential(&mut self, potential: P) -> &mut Self {
        assert!(self.main_potential.size() == potential.size());

        self.potentials.push(potential);

        self
    }
}

impl<P: SubPotential> Potential for Composite<P> {
    type Space = <P as Potential>::Space;
    
    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        self.main_potential.value_inplace(r, value);

        for potential in &self.potentials {
            potential.value_add(r, value);
        }
    }
    
    fn size(&self) -> usize {
        self.main_potential.size()
    }
}

impl<P: SubPotential> SubPotential for Composite<P> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        self.main_potential.value_add(r, value);
        
        for potential in &self.potentials {
            potential.value_add(r, value);
        }
    }
}
