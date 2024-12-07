use std::time::Instant;

use nalgebra::DMatrix;
use quantum::{params::{particle_factory::create_atom, particles::Particles}, problems_impl, units::{distance_units::Distance, energy_units::{Energy, Kelvin}, Au}, utility::linspace};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::dmatrix_backed::DMatrixRatioNumerov, propagator::MultiStepRule}, observables::s_matrix::HasSMatrix, potentials::{dispersion_potential::Dispersion, gaussian_coupling::GaussianCoupling, multi_coupling::MultiCoupling, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::Potential, potential_factory::create_lj}, utility::AngularSpin};


pub struct ManyChannels;

problems_impl!(ManyChannels, "large number of channels",
    "scattering length static size" => |_| Self::scattering_length()
);

impl ManyChannels {
    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(AngularSpin(0));
        
        particles
    }

    fn potential() -> impl Potential<Space = DMatrix<f64>> {
        const N: usize = 50;

        let wells = linspace(0.0019, 0.0022, N);
        let potentials = wells.iter()
            .map(|well| {
                let mut potential = create_lj(Energy(*well, Au), Distance(9.0, Au));
                potential.add_potential(Dispersion::new(Energy(well / 0.0019 - 1.0, Kelvin).to_au(), 0));

                potential
            })
            .collect();
        
        let couplings_strength = linspace(5.0, 15.0, N - 1);
        let couplings = couplings_strength
            .iter()
            .map(|c| GaussianCoupling::new(Energy(*c, Kelvin), 11.0, 2.0))
            .collect();

        let potential = Diagonal::<DMatrix<f64>, _>::from_vec(potentials);
        let coupling = MultiCoupling::<DMatrix<f64>, _>::new_neighboring(couplings);
        PairPotential::new(potential, coupling)
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        let particles = Self::particles();
        let potential = Self::potential();
        
        let id: DMatrix<f64> = DMatrix::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let mut numerov = DMatrixRatioNumerov::new(&potential, &particles, MultiStepRule::default(), boundary);
        let preparation = start.elapsed();
        numerov.propagate_to(1e3);
        let propagation = start.elapsed() - preparation;

        let s_matrix = numerov.data.calculate_s_matrix(0);
        let scattering_length = s_matrix.get_scattering_length();

        let extraction = start.elapsed() - propagation - preparation;

        println!("preparation {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    // fn bound_states() {
    //     println!("Calculating bound states...");
    //     const N: usize = 50;
    //     let mut collision_params = Self::create_dyn_collision_params(N);

    //     let mut bounds = MultiBounds::new(&mut collision_params, (6.5, 1000.0));
    //     let energies = unit_linspace(Energy(-2.0, CmInv), Energy(0.0, CmInv), 100);

    //     let start = Instant::now();
    //     let (bound_diffs, node_counts) =  bounds.bound_diff_dependence(&energies);
    //     let elapsed = start.elapsed();
    //     println!("Elapsed time: {:?}", elapsed.as_secs_f64());

    //     let energies = energies.iter().map(|e| e.value()).collect();
    //     let node_counts = node_counts.into_iter().map(|n| n as f64).collect();
    //     let header = vec![
    //         "energy",
    //         "bound difference",
    //         "node count",
    //     ];
    //     let data = vec![energies, bound_diffs, node_counts];

    //     save_data("many_chan", "bound_diffs", header, data).unwrap();

    //     // let bound_states = vec![0, 1, 3, -1, -2, -5];
    //     // for n in bound_states {
    //     //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
    //     //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

    //     //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));
            
    //     //     let header = vec!["position", "wave function"];
    //     //     let data = vec![rs, wave];
    //     //     save_data("two_chan", &format!("bound_wave_{}", n), header, data).unwrap();
    //     // }
    // }
}
