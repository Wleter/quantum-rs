use abm::{utility::save_spectrum, HifiProblemBuilder};
use clebsch_gordan::half_u32;
use quantum::utility::linspace;

pub struct HifiSingle;

impl HifiSingle {
    pub fn run() {
        let hifi = HifiProblemBuilder::new(half_u32!(1/2), half_u32!(1))
            .with_custom_bohr_magneton(1e-2)
            .with_nuclear_magneton(-2e-4)
            .with_hyperfine_coupling(2.0)
            .build();

        let mag_fields = linspace(0., 1000., 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (values, _) = hifi.states_at(mag_field);

                values.iter().map(|x| x.value()).collect()
            })
            .collect();

        let header = "Magnetic field [a.u]\tEnergies [a.u]";
        save_spectrum(header, "hifi_single", &mag_fields, &values)
            .expect("error while saving results");
    }
}
