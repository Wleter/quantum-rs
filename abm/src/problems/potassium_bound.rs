use abm::{
    utility::save_spectrum, ABMProblemBuilder, ABMVibrational, DoubleHifiProblemBuilder,
    HifiProblemBuilder, Symmetry,
};
use faer::mat;
use quantum::{
    problems_impl,
    units::energy_units::{Energy, GHz, MHz},
    utility::linspace,
};
pub struct PotassiumBound;

problems_impl!(PotassiumBound, "potassium-potassium",
    "hyperfine 39-41" => |_| Self::hifi_39_41(),
    "abm 39-41" => |_| Self::abm_39_41(),
    "hyperfine 39-39" => |_| Self::hifi_39_39(),
    "abm 39-39" => |_| Self::abm_39_39()
);

impl PotassiumBound {
    const HIFI_K39_MHZ: f64 = 230.8595;
    const HIFI_K41_MHZ: f64 = 127.007;
    const GAMMA_I_K39: f64 = 1.989344e-4 * 1e-4;
    const GAMMA_I_K41: f64 = 1.091921e-4 * 1e-4;

    pub fn hifi_39_41() {
        println!("Solving hyperfine for K39 - K41");

        let a_hifi_1 = Energy(Self::HIFI_K39_MHZ, MHz).to_au();
        let a_hifi_2 = Energy(Self::HIFI_K41_MHZ, MHz).to_au();
        let gamma_i1 = Energy(Self::GAMMA_I_K39, MHz).to_au();
        let gamma_i2 = Energy(Self::GAMMA_I_K41, MHz).to_au();

        let first = HifiProblemBuilder::new(1, 3)
            .with_nuclear_magneton(gamma_i1)
            .with_hyperfine_coupling(a_hifi_1);

        let second = HifiProblemBuilder::new(1, 3)
            .with_nuclear_magneton(gamma_i2)
            .with_hyperfine_coupling(a_hifi_2);

        let hifi = DoubleHifiProblemBuilder::new(first, second)
            .with_projection(4)
            .build();

        let mag_fields = linspace(0.0, 600.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (energies, _) = hifi.states_at(mag_field);

                energies.iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [Ghz]";
        save_spectrum(header, "potassium/hifi_39_41", &mag_fields, &values)
            .expect("error in saving the abm");
    }

    pub fn abm_39_41() {
        println!("Solving abm for K39 K41...");

        let a_hifi_1 = Energy(Self::HIFI_K39_MHZ, MHz).to_au();
        let a_hifi_2 = Energy(Self::HIFI_K41_MHZ, MHz).to_au();
        let gamma_i1 = Energy(Self::GAMMA_I_K39, MHz).to_au();
        let gamma_i2 = Energy(Self::GAMMA_I_K41, MHz).to_au();

        let triplet_energies = vec![Energy(-8.33, MHz), Energy(-1282.5, MHz)];
        let singlet_energies = vec![Energy(-32.1, MHz), Energy(-1698.1, MHz)];
        let fc_factors = mat![[0.9180, 0.0463], [0.0895, 0.9674]];

        let first = HifiProblemBuilder::new(1, 3)
            .with_nuclear_magneton(gamma_i1)
            .with_hyperfine_coupling(a_hifi_1);

        let second = HifiProblemBuilder::new(1, 3)
            .with_nuclear_magneton(gamma_i2)
            .with_hyperfine_coupling(a_hifi_2);

        let vibrational = ABMVibrational::new(singlet_energies, triplet_energies, fc_factors);

        let abm_problem = ABMProblemBuilder::new(first, second)
            .with_vibrational(vibrational)
            .with_projection(4)
            .build();

        let mag_fields = linspace(0.0, 600.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (values, _) = abm_problem.states_at(mag_field);

                values.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "potassium/abm_39_41", &mag_fields, &values)
            .expect("error in saving the abm");
    }

    pub fn hifi_39_39() {
        println!("Solving hyperfine for K39 - K39...");

        let a_hifi = Energy(Self::HIFI_K39_MHZ, MHz).to_au();
        let gamma_i = Energy(Self::GAMMA_I_K39, MHz).to_au() * 1e-4;

        let single = HifiProblemBuilder::new(1, 3)
            .with_hyperfine_coupling(a_hifi)
            .with_nuclear_magneton(gamma_i);

        let hifi = DoubleHifiProblemBuilder::new_homo(single, Symmetry::Bosonic)
            .with_projection(4)
            .build();

        let mag_fields = linspace(0.0, 600.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (energies, _) = hifi.states_at(mag_field);

                energies.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "Magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "potassium/hifi_39_39", &mag_fields, &values)
            .expect("Error while saving");
    }

    pub fn abm_39_39() {
        println!("Solving abm for K39 K39...");

        let a_hifi = Energy(Self::HIFI_K39_MHZ, MHz).to_au();
        let gamma_i = Energy(Self::GAMMA_I_K39, MHz).to_au();

        let triplet_energies = vec![Energy(-8.33, MHz), Energy(-1282.5, MHz)];
        let singlet_energies = vec![Energy(-32.1, MHz), Energy(-1698.1, MHz)];
        let fc_factors = mat![[0.9180, 0.0463], [0.0895, 0.9674]];

        let single = HifiProblemBuilder::new(1, 3)
            .with_nuclear_magneton(gamma_i)
            .with_hyperfine_coupling(a_hifi);

        let vibrational = ABMVibrational::new(singlet_energies, triplet_energies, fc_factors);

        let abm_problem = ABMProblemBuilder::new_homo(single, Symmetry::Bosonic)
            .with_vibrational(vibrational)
            .with_projection(4)
            .build();

        let mag_fields = linspace(0.0, 600.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (values, _) = abm_problem.states_at(mag_field);

                values.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "potassium/abm_39_39", &mag_fields, &values)
            .expect("error in saving the abm");
    }
}
