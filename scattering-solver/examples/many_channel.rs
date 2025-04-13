use std::time::Instant;

use faer::Mat;
use indicatif::ParallelProgressIterator;
use quantum::{
    params::{particle_factory::create_atom, particles::Particles},
    problem_selector::{get_args, ProblemSelector},
    problems_impl,
    units::{
        distance_units::Distance, energy_units::{Energy, Kelvin}, Au, GHz
    },
    utility::linspace,
};
use rayon::prelude::*;
use scattering_solver::{
    boundary::{Asymptotic, Boundary, Direction}, log_derivatives::johnson::Johnson, numerovs::{
        multi_numerov::MultiRNumerov, propagator_watcher::PropagatorLogging, LocalWavelengthStepRule
    }, observables::bound_states::BoundProblemBuilder, potentials::{
        dispersion_potential::Dispersion,
        multi_coupling::MultiCoupling,
        multi_diag_potential::Diagonal,
        pair_potential::PairPotential,
        potential::{MatPotential, Potential},
        potential_factory::create_lj,
    }, propagator::{CoupledEquation, Propagator}, utility::{save_data, save_spectrum, AngMomentum}
};

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "large number of channels",
    "scattering length" => |_| Self::scattering_length(),
    "bound states" => |_| Self::bound_states(),
);

const N: usize = 50;

impl Problems {
    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let wells = linspace(0.0019, 0.0022, N);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(Asymptotic {
            centrifugal: vec![AngMomentum(0); N],
            entrance: 0,
            channel_energies: wells
                .iter()
                .map(|well| Energy(well / 0.0019 - 1.0, Kelvin).to_au())
                .collect(),
            channel_states: Mat::identity(N, N),
        });

        particles
    }

    fn potential() -> impl MatPotential {
        let wells = linspace(0.0019, 0.0022, N);
        let potentials = wells
            .iter()
            .map(|well| {
                let mut potential = create_lj(Energy(*well, Au), Distance(9.0, Au));
                potential.add_potential(Dispersion::new(
                    Energy(well / 0.0019 - 1.0, Kelvin).to_au(),
                    0,
                ));

                potential
            })
            .collect();

        let couplings = wells
            .iter()
            .skip(1)
            .map(|well| create_lj(Energy(well / 10., Au), Distance(9., Au)))
            .collect();

        let potential = Diagonal::<Mat<f64>, _>::from_vec(potentials);
        let coupling = MultiCoupling::<Mat<f64>, _>::new_neighboring(couplings);
        PairPotential::new(potential, coupling)
    }

    fn scattering_length() {
        let start = Instant::now();

        let particles = Self::particles();
        let potential = Self::potential();

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let eq = CoupledEquation::from_particles(&potential, &particles);

        let step_rule = LocalWavelengthStepRule::default();
        let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
        let preparation = start.elapsed();

        numerov.propagate_to_with(1e3, &mut PropagatorLogging::default());
        let propagation = start.elapsed() - preparation;

        let s_matrix = numerov.s_matrix();
        let scattering_length = s_matrix.get_scattering_length();

        let extraction = start.elapsed() - propagation - preparation;

        println!("preparation {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn bound_states() {
        let particles = Self::particles();
        let potential = Self::potential();

        let bound_problem = BoundProblemBuilder::new(&particles, &potential)
            .with_propagation(LocalWavelengthStepRule::new(1e-4, 10., 500.), Johnson)
            .with_range(6.5, 20., 500.)
            .build();

        let energies = linspace(Energy(-100.0, GHz).to_au(), Energy(0.0, GHz).to_au(), 1000);
        let data: Vec<(u64, Vec<f64>)> = energies
            .par_iter()
            .progress()
            .map(|&energy| {
                let bound_mismatch = bound_problem.bound_mismatch(Energy(energy, Au));

                (bound_mismatch.nodes, bound_mismatch.matching_eigenvalues)
            })
            .collect();

        let node_counts = data.iter().map(|x| x.0 as f64).collect();
        let mismatch: Vec<Vec<f64>> = data.into_iter().map(|x| x.1).collect();

        let energies = energies
            .into_iter()
            .map(|x| Energy(x, Au).to(GHz).value())
            .collect::<Vec<f64>>();

        let header = "energy\tnode_count";
        let data = vec![energies.clone(), node_counts];

        save_data("many_chan/node_count", header, &data).unwrap();
        save_spectrum("many_chan/bound_mismatch", "energy\tmismatches", &energies, &mismatch).unwrap()
    }
}
