use abm::utility::diagonalize;
use faer::{unzipped, zipped, Mat};
use scattering_solver::utility::AngMomentum;

#[derive(Debug)]
pub struct AngularBlock {
    pub ang_momentum: AngMomentum,
    pub field_inv: Mat<f64>,
    pub field_prop: Mat<f64>
}

impl AngularBlock {
    pub fn size(&self) -> usize {
        self.field_inv.nrows()
    }
}

#[derive(Debug)]
pub struct AngularBlocks(pub Vec<AngularBlock>);

impl AngularBlocks {
    pub fn diagonalize(&self, field: f64) -> (Vec<f64>, Mat<f64>) {
        let n = self.size();
        let mut energies = Vec::with_capacity(n);
        let mut eigenstates = Mat::<f64>::zeros(n, n);

        let mut block_index = 0;
        for block in &self.0 {
            let internal = &block.field_inv + field * &block.field_prop;
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