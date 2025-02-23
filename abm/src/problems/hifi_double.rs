use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder, utility::save_spectrum};
use clebsch_gordan::{hi32, hu32};
use quantum::utility::linspace;

pub struct HifiDouble;

impl HifiDouble {
    pub fn run() {
        let first = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
            .with_custom_bohr_magneton(1e-2)
            .with_nuclear_magneton(-2.2e-4)
            .with_hyperfine_coupling(2.0);

        let second = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(3 / 2))
            .with_custom_bohr_magneton(1e-2)
            .with_nuclear_magneton(-1.8e-4)
            .with_hyperfine_coupling(2.2);

        let hifi = DoubleHifiProblemBuilder::new(first, second)
            .with_projection(hi32!(0))
            .build();

        let mag_fields = linspace(0.0, 1000.0, 1000);
        let values: Vec<Vec<f64>> = mag_fields
            .iter()
            .map(|&mag_field| {
                let (values, _) = hifi.states_at(mag_field);

                values.iter().map(|x| x.value()).collect()
            })
            .collect();

        let header = "Magnetic field [a.u]\tEnergies [a.u]";
        save_spectrum(header, "hifi_double", &mag_fields, &values)
            .expect("error while saving results");
    }
}
