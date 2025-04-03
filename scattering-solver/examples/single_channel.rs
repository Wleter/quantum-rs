use core::f64;
use std::time::Instant;

use num::Complex;
use quantum::{
    params::{particle_factory::create_atom, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{
        distance_units::Distance, energy_units::{Energy, Kelvin}, mass_units::Mass, Au, GHz
    }, utility::linspace
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    numerovs::{
        propagator_watcher::{
            ManyPropagatorWatcher, PropagatorLogging, Sampling, ScatteringVsDistance, WaveStorage,
        }, single_numerov::SingleRNumerov, LocalWavelengthStepRule
    },
    potentials::{
        potential::{Potential, SimplePotential},
        potential_factory::create_lj,
    },
    propagator::{Propagator, SingleEquation},
    utility::{save_data, AngMomentum},
};

pub fn main() {
    Problems::select(&mut get_args());
}


pub struct Problems {}

problems_impl!(Problems, "single channel",
    "wave function" => |_| Self::wave_function(),
    "scattering length" => |_| Self::scattering_length(),
    "propagation distance" => |_| Self::propagation_distance(),
    "mass scaling" => |_| Self::mass_scaling(),
    "bound states" => |_| Self::bound_states(),
);

impl Problems {
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
        let particles = Self::particles();
        let potential = Self::potential();

        let eq = SingleEquation::from_particles(&potential, &particles);

        let mut numerov = SingleRNumerov::new(
            eq,
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
            LocalWavelengthStepRule::default(),
        );
        let mut wave_storage = WaveStorage::new(Sampling::default(), 1e-50, 500);
        let mut numerov_logging = PropagatorLogging::default();

        let mut watchers =
            ManyPropagatorWatcher::new(vec![&mut wave_storage, &mut numerov_logging]);

        numerov.propagate_to_with(100., &mut watchers);

        let potential_values: Vec<f64> = wave_storage
            .rs
            .iter()
            .map(|&r| potential.value(r))
            .collect();

        let header = "position\twave function\tpotential";
        let data = vec![wave_storage.rs, wave_storage.waves, potential_values];
        save_data("single_chan/wave_function", header, &data).unwrap();
    }

    fn scattering_length() {
        let particles = Self::particles();
        let potential = Self::potential();
        let eq = SingleEquation::from_particles(&potential, &particles);

        let mut numerov = SingleRNumerov::new(
            eq,
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
            LocalWavelengthStepRule::default(),
        );

        let start = Instant::now();
        numerov.propagate_to(1000.0);
        let propagation = start.elapsed();

        let s_matrix = numerov.solution().s_matrix(numerov.equation());
        let scattering_length = s_matrix.get_scattering_length();

        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn propagation_distance() {
        let particles = Self::particles();
        let potential = Self::potential();

        let eq = SingleEquation::from_particles(&potential, &particles);

        let mut numerov = SingleRNumerov::new(
            eq,
            Boundary::new(6.5, Direction::Outwards, (1.001, 1.002)),
            LocalWavelengthStepRule::default(),
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

                let eq = SingleEquation::from_particles(&potential, &particles);
                let mut numerov =
                    SingleRNumerov::new(eq, boundary.clone(), LocalWavelengthStepRule::default());

                numerov.propagate_to(1e4);

                numerov.s_matrix().get_scattering_length()
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
        let data: Vec<f64> = energies.iter()
            .map(|&energy| {
                let mut particles = particles.clone();
                particles.insert(Energy(energy, Au));

                let eq = SingleEquation::from_particles(&potential, &particles);
                let boundary = Boundary::new_vanishing(6.5, Direction::Outwards);
                let step_rule = LocalWavelengthStepRule::new(1e-4, 10., 500.);

                let mut numerov = SingleRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(500.).nodes as f64
            })
            .collect();

        // let bound_diffs = data.iter().map(|n| n.diff as f64).collect();
        // let node_counts = data.iter().map(|n| n as f64).collect();
        let energies = energies.into_iter().map(|x| Energy(x, Au).to(GHz).value()).collect();

        let header = "energy\tnode_count";
        let data = vec![energies, data];

        save_data("single_chan/node_count", header, &data).unwrap();

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
