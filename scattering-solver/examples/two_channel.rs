use std::time::Instant;

use faer::Mat;
use num::Complex;
use quantum::{
    params::{particle_factory::create_atom, particles::Particles},
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::{
        Au, GHz,
        distance_units::Distance,
        energy_units::{Energy, Kelvin},
        mass_units::Mass,
    },
    utility::linspace,
};
use scattering_solver::{
    boundary::{Asymptotic, Boundary, Direction},
    log_derivatives::johnson::JohnsonLogDerivative,
    numerovs::{
        LocalWavelengthStepRule,
        multi_numerov::MultiRNumerov,
        propagator_watcher::{ManyPropagatorWatcher, PropagatorLogging, Sampling, WaveStorage},
    },
    potentials::{
        dispersion_potential::Dispersion,
        gaussian_coupling::GaussianCoupling,
        multi_coupling::MultiCoupling,
        multi_diag_potential::Diagonal,
        pair_potential::PairPotential,
        potential::{MatPotential, Potential},
        potential_factory::create_lj,
    },
    propagator::{CoupledEquation, Propagator},
    utility::{AngMomentum, save_data},
};

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "two channel",
    "wave function" => |_| Self::wave_function(),
    "scattering length" => |_| Self::scattering_length(),
    "mass scaling" => |_| Self::mass_scaling(),
    "bound states" => |_| Self::bound_states(),
    "testing" => |_| Self::testing()
);

impl Problems {
    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(Asymptotic {
            centrifugal: vec![AngMomentum(0); 2],
            entrance: 0,
            channel_energies: vec![0., Energy(0.0021, Kelvin).to_au()],
            channel_states: Mat::identity(2, 2),
        });

        particles
    }

    fn potential() -> impl MatPotential {
        let potential_lj1 = create_lj(Energy(0.002, Au), Distance(9., Au));
        let mut potential_lj2 = create_lj(Energy(0.0021, Au), Distance(8.9, Au));
        potential_lj2.add_potential(Dispersion::new(Energy(1., Kelvin).to_au(), 0));

        let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);

        let potential = Diagonal::<Mat<f64>, _>::from_vec(vec![potential_lj1, potential_lj2]);
        let coupling = MultiCoupling::<Mat<f64>, _>::new_neighboring(vec![coupling]);

        PairPotential::new(potential, coupling)
    }

    fn wave_function() {
        let particles = Self::particles();
        let potential = Self::potential();

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        // let boundary = Boundary::new_multi_vanishing(6.5, Direction::Outwards, potential.size());

        let eq = CoupledEquation::from_particles(&potential, &particles);

        let boundary = Boundary::new_exponential_vanishing(500., &eq);
        // let boundary = Boundary::new_multi_vanishing(6.5, Direction::Outwards, potential.size());

        let step_rule = LocalWavelengthStepRule::new(1e-4, 10., 500.);
        let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

        let mut wave_storage = WaveStorage::new(Sampling::Uniform, 1e-50 * id, 500);
        let mut numerov_logging = PropagatorLogging::default();

        let mut watchers =
            ManyPropagatorWatcher::new(vec![&mut wave_storage, &mut numerov_logging]);

        numerov.propagate_to_with(6.5, &mut watchers);

        let chan1 = wave_storage.waves.iter().map(|wave| wave[(0, 0)]).collect();
        let chan2 = wave_storage.waves.iter().map(|wave| wave[(0, 1)]).collect();
        let nodes = wave_storage
            .nodes
            .iter()
            .map(|&nodes| nodes as f64)
            .collect();

        let header = "position\tchannel_1\tchannel_2\tnodes";
        let data = vec![wave_storage.rs, chan1, chan2, nodes];
        save_data("two_chan/wave_function", header, &data).unwrap();
    }

    fn scattering_length() {
        let particles = Self::particles();
        let potential = Self::potential();

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let eq = CoupledEquation::from_particles(&potential, &particles);
        let mut numerov =
            JohnsonLogDerivative::new(eq, boundary, LocalWavelengthStepRule::default());

        let start = Instant::now();
        numerov.propagate_to(1e3);
        let propagation = start.elapsed();

        let s_matrix = numerov.s_matrix();
        let scattering_length = s_matrix.get_scattering_length();

        let extraction = start.elapsed() - propagation;

        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn mass_scaling() {
        let mut particles = Self::particles();
        let potential = Self::potential();
        let scalings = linspace(0.8, 1.2, 200);

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));
        let mass = particles.red_mass();

        let s_lengths: Vec<Complex<f64>> = scalings
            .iter()
            .map(|scaling| {
                particles.get_mut::<Mass<Au>>().unwrap().0 = mass * scaling;

                let eq = CoupledEquation::from_particles(&potential, &particles);
                let mut numerov =
                    MultiRNumerov::new(eq, boundary.clone(), LocalWavelengthStepRule::default());

                numerov.propagate_to(1e3);

                let s_matrix = numerov.s_matrix();
                s_matrix.get_scattering_length()
            })
            .collect();

        let scat_re = s_lengths.iter().map(|s| s.re).collect();
        let scat_im = s_lengths.iter().map(|s| s.im).collect();

        let header = "mass scale factor\t\
            scattering length real\t\
            scattering length imag";
        let data = vec![scalings, scat_re, scat_im];

        save_data("two_chan/mass_scaling", header, &data).unwrap();
    }

    fn bound_states() {
        let particles = Self::particles();
        let potential = Self::potential();

        let energies = linspace(Energy(-100.0, GHz).to_au(), Energy(0.0, GHz).to_au(), 1000);
        let data: Vec<f64> = energies
            .iter()
            .map(|&energy| {
                let mut particles = particles.clone();
                particles.insert(Energy(energy, Au));

                let eq = CoupledEquation::from_particles(&potential, &particles);
                let boundary = Boundary::new_exponential_vanishing(500., &eq);

                let step_rule = LocalWavelengthStepRule::new(1e-4, 10., 500.);

                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

                numerov.propagate_to(6.5).nodes as f64
            })
            .collect();

        // let bound_diffs = data.iter().map(|n| n.diff as f64).collect();
        // let node_counts = data.iter().map(|n| n as f64).collect();
        let energies = energies
            .into_iter()
            .map(|x| Energy(x, Au).to(GHz).value())
            .collect();

        let header = "energy\tnode_count";
        let data = vec![energies, data];

        save_data("two_chan/node_count", header, &data).unwrap();

        // let bound_states = vec![0, 1, 3, -1, -2, -5];
        // for n in bound_states {
        //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
        //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

        //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));

        //     let header = vec!["position", "wave function"];
        //     let data = vec![rs, wave];
        //     save_data("two_chan", &format!("bound_wave_{}", n), header, data).unwrap();
        // }
    }

    fn testing() {}
}
