use abm::utility::diagonalize;
use faer::{unzipped, zipped, Mat};
use scattering_solver::utility::AngMomentum;

#[derive(Debug)]
pub struct AngularBlock {
    pub ang_momentum: AngMomentum,
    field_inv: Vec<Mat<f64>>,
    field_prop: Vec<Mat<f64>>  // todo! change so that each term has a label identifying term
}

impl AngularBlock {
    pub fn new(ang_momentum: AngMomentum, field_inv: Vec<Mat<f64>>, field_prop: Vec<Mat<f64>>) -> Self {
        assert!(!field_inv.is_empty());
        assert!(!field_prop.is_empty());
        let n = field_inv[0].nrows();

        for mat in &field_inv {
            assert!(n == mat.nrows(), "wrong size of field invariant matrices");
            assert!(n == mat.ncols(), "wrong size of field invariant matrices");
        }

        for mat in &field_prop {
            assert!(n == mat.nrows(), "wrong size of field proportional matrices");
            assert!(n == mat.ncols(), "wrong size of field proportional matrices");
        }

        Self {
            ang_momentum,
            field_inv,
            field_prop
        }
    }

    pub fn field_inv(&self) -> Mat<f64> {
        let n = self.field_inv[0].nrows();
        let mut field_inv = Mat::zeros(n, n);
        for mat in &self.field_inv {
            field_inv += mat
        }

        field_inv
    }

    pub fn field_prop(&self) -> Mat<f64> {
        let n = self.field_prop[0].nrows();
        let mut field_prop = Mat::zeros(n, n);
        for mat in &self.field_prop {
            field_prop += mat
        }

        field_prop
    }

    pub fn field_inv_matrices(&self) -> &[Mat<f64>] {
        &self.field_inv
    }

    pub fn field_prop_matrices(&self) -> &[Mat<f64>] {
        &self.field_prop
    }

    pub fn size(&self) -> usize {
        self.field_inv[0].nrows()
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
            let internal = &block.field_inv() + field * &block.field_prop();
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