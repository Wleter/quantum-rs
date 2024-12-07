use std::time::Instant;

use faer::Mat;
use num::Complex;
use quantum::{params::{particle_factory::create_atom, particles::Particles}, problems_impl, units::{distance_units::Distance, energy_units::{Energy, Kelvin}, mass_units::Mass, Au}, utility::linspace};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::faer_backed::FaerRatioNumerov, numerov_modifier::{Sampling, WaveStorage}, propagator::MultiStepRule}, observables::s_matrix::HasSMatrix, potentials::{dispersion_potential::Dispersion, gaussian_coupling::GaussianCoupling, multi_coupling::MultiCoupling, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::Potential, potential_factory::create_lj}, utility::{save_data, AngularSpin}};


pub struct TwoChannel;

problems_impl!(TwoChannel, "two channel", 
    "wave function" => |_| Self::wave_function(),
    "scattering length" => |_| Self::scattering_length(),
    "mass scaling" => |_| Self::mass_scaling()
);
// "bound states" => |_| Self::bound_states()

impl TwoChannel {
    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(AngularSpin(0));

        particles
    }

    fn potential() -> impl Potential<Space = Mat<f64>> {
        let potential_lj1 = create_lj(Energy(0.002, Au), Distance(9., Au));
        let mut potential_lj2 = create_lj(Energy(0.0021, Au), Distance(8.9, Au));
        potential_lj2.add_potential(Dispersion::new(Energy(1., Kelvin).to_au(), 0));

        let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);

        let potential = Diagonal::<Mat<f64>, _>::from_vec(vec![potential_lj1, potential_lj2]);
        let coupling = MultiCoupling::<Mat<f64>, _>::new_neighboring(vec![coupling]);

        PairPotential::new(potential, coupling)
    }

    fn wave_function() {
        println!("Calculating wave function...");

        let start = Instant::now();

        let particles = Self::particles();
        let potential = Self::potential();

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let mut numerov = FaerRatioNumerov::new(&potential, &particles, MultiStepRule::default(), boundary);
        let mut wave_storage = WaveStorage::new(Sampling::default(), 1e-50 * id, 500);

        let preparation = start.elapsed();
        numerov.propagate_to_with(100., &mut wave_storage);
        let propagation = start.elapsed() - preparation;

        let chan1 = wave_storage.waves.iter().map(|wave| wave[(0, 0)]).collect();
        let chan2 = wave_storage.waves.iter().map(|wave| wave[(0, 1)]).collect();

        let header = "position\tchannel_1\tchannel_2";
        let data = vec![wave_storage.rs, chan1, chan2];
        save_data("two_chan/wave_function", header, &data).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        println!("Calculating scattering length...");

        let particles = Self::particles();
        let potential = Self::potential();

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));

        let mut numerov = FaerRatioNumerov::new(&potential, &particles, MultiStepRule::default(), boundary);
        let start = Instant::now();
        numerov.propagate_to(1e3);
        let propagation = start.elapsed();

        let s_matrix = numerov.data.calculate_s_matrix(0);
        let scattering_length = s_matrix.get_scattering_length();

        let extraction = start.elapsed() - propagation;

        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn mass_scaling() {
        println!("Calculating scattering length vs mass scaling...");

        let mut particles = Self::particles();
        let potential = Self::potential();
        let scalings = linspace(0.8, 1.2, 200);
        let entrance = 0;

        let id: Mat<f64> = Mat::identity(potential.size(), potential.size());
        let boundary = Boundary::new(6.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));
        let mass = particles.red_mass();

        let s_lengths: Vec<Complex<f64>> = scalings.iter()
            .map(|scaling| {
                particles.get_mut::<Mass<Au>>().unwrap().0 = mass * scaling;
    
                let mut numerov = FaerRatioNumerov::new(&potential, &particles, MultiStepRule::default(), boundary.clone());
                numerov.propagate_to(1e3);
        
                let s_matrix = numerov.data.calculate_s_matrix(entrance);
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

    // fn bound_states() {
    //     println!("Calculating bound states...");

    //     let mut collision_params = Self::create_dyn_collision_params();
    //     let mut bounds = MultiBounds::new(&mut collision_params, (6.5, 1000.0));

    //     let energies = unit_linspace(Energy(-2.0, CmInv), Energy(0.0, CmInv), 500);
    //     let (bound_diffs, node_counts) =  bounds.bound_diff_dependence(&energies);

    //     let energies = energies.iter().map(|e| e.value()).collect();
    //     let node_counts = node_counts.into_iter().map(|n| n as f64).collect();
    //     let header = vec![
    //         "energy",
    //         "bound difference",
    //         "node count",
    //     ];
    //     let data = vec![energies, bound_diffs, node_counts];

    //     save_data("two_chan", "bound_diffs", header, data).unwrap();

        // let bound_states = vec![0, 1, 3, -1, -2, -5];
        // for n in bound_states {
        //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
        //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

        //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));
            
        //     let header = vec!["position", "wave function"];
        //     let data = vec![rs, wave];
        //     save_data("two_chan", &format!("bound_wave_{}", n), header, data).unwrap();
        // }
    // }
}
