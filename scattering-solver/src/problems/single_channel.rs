use core::f64;
use std::time::Instant;

use num::Complex;
use quantum::{
    params::{particle_factory::create_atom, particles::Particles},
    problems_impl,
    units::{
        distance_units::Distance, energy_units::{Energy, GHz, Kelvin}, mass_units::Mass, Au
    },
    utility::linspace,
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    numerovs::{
        bound_numerov::{BoundDiff, SingleBoundRatioNumerov}, numerov_modifier::{Sampling, ScatteringVsDistance, WaveStorage}, propagator::MultiStepRule, single_numerov::SingleRatioNumerov
    },
    potentials::{potential::Potential, potential_factory::create_lj},
    utility::{save_data, AngMomentum},
};

pub struct SingleChannel {}

problems_impl!(SingleChannel, "single channel",
    "wave function" => |_| Self::wave_function(),
    "scattering length" => |_| Self::scattering_length(),
    "propagation distance" => |_| Self::propagation_distance(),
    "mass scaling" => |_| Self::mass_scaling(),
    "bound states" => |_| Self::bound_states(),
);

impl SingleChannel {
    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);
        let spin = AngMomentum(0);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(spin);

        particles
    }

    fn potential() -> impl Potential<Space = f64> {
        create_lj(Energy(0.002, Au), Distance(9.0, Au))
    }

    fn wave_function() {
        let start = Instant::now();

        let particles = Self::particles();
        let potential = Self::potential();

        let mut numerov = SingleRatioNumerov::new(
            &potential,
            &particles,
            MultiStepRule::default(),
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
        );
        let mut wave_storage = WaveStorage::new(Sampling::default(), 1e-50, 500);

        let preparation = start.elapsed();
        numerov.propagate_to_with(100., &mut wave_storage);
        let propagation = start.elapsed() - preparation;

        let potential_values: Vec<f64> = wave_storage
            .rs
            .iter()
            .map(|&r| numerov.data.potential_value(r))
            .collect();

        let header = "position\twave function\tpotential";
        let data = vec![wave_storage.rs, wave_storage.waves, potential_values];
        save_data("single_chan/wave_function", header, &data).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        let particles = Self::particles();
        let potential = Self::potential();

        let mut numerov = SingleRatioNumerov::new(
            &potential,
            &particles,
            MultiStepRule::default(),
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
        );

        let start = Instant::now();
        numerov.propagate_to(1000.0);
        let propagation = start.elapsed();

        let s_matrix = numerov.data.calculate_s_matrix().unwrap();
        let scattering_length = s_matrix.get_scattering_length();

        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn propagation_distance() {
        let particles = Self::particles();
        let potential = Self::potential();

        let mut numerov = SingleRatioNumerov::new(
            &potential,
            &particles,
            MultiStepRule::default(),
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
        );
        let mut scatterings = ScatteringVsDistance::new(120., 1000);

        numerov.propagate_to_with(10000., &mut scatterings);

        let s_lengths: Vec<Complex<f64>> = scatterings
            .s_matrices
            .iter()
            .map(|s| s.get_scattering_length())
            .collect();

        let scat_re = s_lengths.iter().map(|s| s.re).collect();
        let scat_im = s_lengths.iter().map(|s| s.im).collect();
        let header = "distance\t\
            scattering length real\t\
            scattering length imag";
        let data = vec![scatterings.distances, scat_re, scat_im];

        save_data("single_chan/propagation_distance", header, &data).unwrap();
    }

    fn mass_scaling() {
        let mut particles = Self::particles();
        let potential = Self::potential();

        let scalings = linspace(0.8, 1.2, 1000);

        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001, 1.002));
        let mass = particles.red_mass();

        let s_lengths: Vec<Complex<f64>> = scalings
            .iter()
            .map(|scaling| {
                particles.get_mut::<Mass<Au>>().unwrap().0 = mass * scaling;

                let mut numerov = SingleRatioNumerov::new(
                    &potential,
                    &particles,
                    MultiStepRule::default(),
                    boundary.clone(),
                );
                numerov.propagate_to(1e4);

                let s_matrix = numerov.data.calculate_s_matrix().unwrap();
                s_matrix.get_scattering_length()
            })
            .collect();

        let scat_re = s_lengths.iter().map(|s| s.re).collect();
        let scat_im = s_lengths.iter().map(|s| s.im).collect();

        let header = "mass scale factor\t\
            scattering length real\t\
            scattering length imag";
        let data = vec![scalings, scat_re, scat_im];

        save_data("single_chan/mass_scaling", header, &data).unwrap();
    }

    fn bound_states() {
        let particles = Self::particles();
        let potential = Self::potential();

        let energies = linspace(Energy(-100.0, GHz).to_au(), Energy(0.0, GHz).to_au(), 1000);
        let data: Vec<BoundDiff> = energies.iter()
            .map(|&energy| {
                let mut particles = particles.clone();
                particles.insert(Energy(energy, Au));

                SingleBoundRatioNumerov::new(MultiStepRule::new(4e-3, 10., 400.))
                    .bound_diff(&potential, &particles, (6.5, 1000.))
            })
            .collect();

        let bound_diffs = data.iter().map(|n| n.diff as f64).collect();
        let node_counts = data.iter().map(|n| n.nodes as f64).collect();
        let energies = energies.into_iter().map(|x| Energy(x, Au).to(GHz).value()).collect();

        let header = "energy\tbound_diff\tnode_count";
        let data = vec![energies, bound_diffs, node_counts];

        save_data("single_chan/bound_diffs", header, &data).unwrap();

        // let bound_states = vec![0, 1, 3, -1, -2, -5];
        // for n in bound_states {
        //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
        //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

        //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));

        //     let header = vec!["position", "wave function"];
        //     let data = vec![rs, wave];
        //     save_data("single_chan", &format!("bound_wave_{}", n), header, data).unwrap();
        // }
    }
}
