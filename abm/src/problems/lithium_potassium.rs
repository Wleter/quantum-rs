use abm::{utility::save_spectrum, ABMProblemBuilder, ABMVibrational, HifiProblemBuilder};
use faer::mat;
use quantum::{
    problems_impl,
    units::energy_units::{Energy, GHz, MHz},
    utility::linspace,
};

pub struct LithiumPotassium;

problems_impl!(LithiumPotassium, "Li6 - K40",
    "hyperfine" => |_| Self::hifi(),
    "abm" => |_| Self::abm()
);

impl LithiumPotassium {
    const HIFI_LI6_MHZ: f64 = 228.2 / 1.5;
    const HIFI_K40_MHZ: f64 = -1285.8 / 4.5;

    pub fn hifi() {
        println!("Solving hyperfine for Li6 - K40...");

        // ---------- Li6 ----------
        let a_hifi = Energy(Self::HIFI_LI6_MHZ, MHz).to_au();

        let hifi_problem = HifiProblemBuilder::new(1, 2)
            .with_hyperfine_coupling(a_hifi)
            .build();

        let mag_fields = linspace(0.0, 1000.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (energies, _) = hifi_problem.states_at(mag_field);

                energies.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "Magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "Li_K/hifi_Li6", &mag_fields, &values).expect("Error while saving");

        // ---------- K40 ----------
        let a_hifi = Energy(Self::HIFI_K40_MHZ, MHz).to_au();

        let hifi_problem = HifiProblemBuilder::new(1, 8)
            .with_hyperfine_coupling(a_hifi)
            .build();

        let mag_fields = linspace(0.0, 1000.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (energies, _) = hifi_problem.states_at(mag_field);

                energies.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "Magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "Li_K/hifi_K40", &mag_fields, &values).expect("Error while saving");
    }

    pub fn abm() {
        println!("Solving abm for K39 K41...");

        let a_hifi_1 = Energy(Self::HIFI_LI6_MHZ, MHz).to_au();
        let a_hifi_2 = Energy(Self::HIFI_K40_MHZ, MHz).to_au();

        let lithium = HifiProblemBuilder::new(1, 2).with_hyperfine_coupling(a_hifi_1);

        let potassium = HifiProblemBuilder::new(1, 8).with_hyperfine_coupling(a_hifi_2);

        let triplet_state = vec![Energy(-427.44, MHz)];
        let singlet_state = vec![Energy(-720.76, MHz)];
        let fc_factor = mat![[0.979]];

        let vibrational = ABMVibrational::new(singlet_state, triplet_state, fc_factor);

        let abm_problem = ABMProblemBuilder::new(lithium, potassium)
            .with_vibrational(vibrational)
            .with_projection(-6)
            .build();

        let mag_fields = linspace(0.0, 400.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (energies, _) = abm_problem.states_at(mag_field);

                energies.into_iter().map(|x| x.to(GHz).value()).collect()
            })
            .collect();

        let header = "Magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "Li_K/abm", &mag_fields, &values).expect("Error while saving");
    }
}
