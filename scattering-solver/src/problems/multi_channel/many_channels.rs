use std::time::Instant;

use faer::Mat;
use quantum::{
    params::{particle_factory::create_atom, particles::Particles},
    problems_impl,
    units::{
        Au,
        distance_units::Distance,
        energy_units::{Energy, Kelvin},
    },
    utility::linspace,
};
use scattering_solver::{
    boundary::{Asymptotic, Boundary, Direction},
    numerovs::{
        LocalWavelengthStepRule, multi_numerov::MultiRNumerov, numerov_modifier::NumerovLogging,
    },
    potentials::{
        dispersion_potential::Dispersion,
        multi_coupling::MultiCoupling,
        multi_diag_potential::Diagonal,
        pair_potential::PairPotential,
        potential::{MatPotential, Potential},
        potential_factory::create_lj,
    },
    propagator::{CoupledEquation, Propagator},
    utility::AngMomentum,
};
pub struct ManyChannels;

problems_impl!(ManyChannels, "large number of channels",
    "scattering length" => |_| Self::scattering_length(),
    "bound states" => |_| Self::bound_states(),
);

const N: usize = 50;

impl ManyChannels {
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
            .map(|well| create_lj(Energy(well / 10., Au), Distance(6., Au)))
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

        numerov.propagate_to_with(1e3, &mut NumerovLogging::default());
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
        // let particles = Self::particles();
        // let potential = Self::potential();

        // let energies = linspace(Energy(-100.0, GHz).to_au(), Energy(0.0, GHz).to_au(), 100);
        // let data: Vec<BoundDiff> = energies.iter()
        //     .map(|&energy| {
        //         let mut particles = particles.clone();
        //         particles.insert(Energy(energy, Au));

        //         MultiBoundRatioNumerov::new(MultiStepRule::new(1e-4, 10., 500.))
        //             .bound_diff(&potential, &particles, (6.5, 20.0, 500.))
        //     })
        //     .collect();

        //     let bound_diffs = data.iter().map(|n| n.diff as f64).collect();
        //     let node_counts = data.iter().map(|n| n.nodes as f64).collect();
        //     let energies = energies.into_iter().map(|x| Energy(x, Au).to(GHz).value()).collect();

        //     let header = "energy\tbound_diff\tnode_count";
        //     let data = vec![energies, bound_diffs, node_counts];

        // save_data("many_chan/bound_diffs", header, &data).unwrap();

        // let bound_states = vec![0, 1, 3, -1, -2, -5];
        // for n in bound_states {
        //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
        //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

        //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));

        //     let header = vec!["position", "wave function"];
        //     let data = vec![rs, wave];
        //     save_data("many_chan", &format!("bound_wave_{}", n), header, data).unwrap();
        // }
    }
}
