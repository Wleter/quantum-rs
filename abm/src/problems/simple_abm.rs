use abm::{
    ABMProblemBuilder, ABMVibrational, HifiProblemBuilder, Symmetry, utility::save_spectrum,
};
use clebsch_gordan::{hi32, hu32};
use faer::mat;
use quantum::{
    units::{Au, energy_units::Energy},
    utility::linspace,
};

pub struct SimpleABM;

impl SimpleABM {
    pub fn run() {
        let a_hifi = 1.0;
        let gamma_e = 2e-2;
        let gamma_i = -1.2e-5;

        let single = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1))
            .with_custom_bohr_magneton(gamma_e)
            .with_nuclear_magneton(gamma_i)
            .with_hyperfine_coupling(a_hifi);

        let triplet_state = vec![Energy(-5.0, Au)];
        let singlet_state = vec![Energy(-10.0, Au)];
        let gordon = mat![[0.3]];
        let bounds = ABMVibrational::new(singlet_state, triplet_state, gordon);

        let abm_problem = ABMProblemBuilder::new_homo(single, Symmetry::Fermionic)
            .with_vibrational(bounds)
            .with_projection(hi32!(0))
            .build();

        let mag_fields = linspace(0., 500., 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (values, _) = abm_problem.states_at(mag_field);

                values.into_iter().map(|x| x.value()).collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [a.u.]";
        save_spectrum(header, "simple_abm", &mag_fields, &values).expect("error in saving the abm");
    }
}
