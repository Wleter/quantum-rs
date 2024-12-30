use abm::utility::diagonalize;
use faer::{unzipped, zipped, Mat};
use scattering_solver::{boundary::Asymptotic, potentials::potential::Potential, utility::AngMomentum};

pub mod alkali_atoms;
pub mod alkali_rotor_atom;
pub mod utility;
pub mod rotor_atom;
pub mod potential_interpolation;
pub mod operators;
pub mod alkali_rotor;

pub struct ScatteringProblem<P: Potential<Space = Mat<f64>>> {
    pub potential: P,
    pub asymptotic: Asymptotic,
}

pub struct AngularBlock {
    pub ang_momentum: AngMomentum,
    pub mag_inv: Mat<f64>,
    pub mag_prop: Mat<f64>
}

impl AngularBlock {
    pub fn size(&self) -> usize {
        self.mag_inv.nrows()
    }
}

pub struct AngularBlocks(pub Vec<AngularBlock>);

impl AngularBlocks {
    pub fn diagonalize(&self, mag_field: f64) -> (Vec<f64>, Mat<f64>) {
        let n = self.size();
        let mut energies = Vec::with_capacity(n);
        let mut eigenstates = Mat::<f64>::zeros(n, n);

        let mut block_index = 0;
        for block in &self.0 {
            let internal = &block.mag_inv + mag_field * &block.mag_prop;
            let n_block = block.size();
    
            let (energies_block, eigenstates_block) = diagonalize(internal.as_ref());

            energies.extend(energies_block);
            let mut sub_matrix = eigenstates.submatrix_mut(block_index, block_index, n_block, n_block);
            zipped!(sub_matrix, eigenstates_block.as_ref())
                .for_each(|unzipped!(s, &e)| *s = e);

            block_index += n_block;
        }

        (energies, eigenstates)
    }

    pub fn angular_states(&self) -> Vec<AngMomentum> {
        let mut states = Vec::with_capacity(self.size());
        for b in &self.0 {
            states.append(&mut vec![b.ang_momentum; b.size()]);
        }

        states
    }

    pub fn size(&self) -> usize {
        self.0.iter().map(|b| b.size()).sum()
    }
}