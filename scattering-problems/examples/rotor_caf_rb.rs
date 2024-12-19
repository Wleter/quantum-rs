use core::f64;
use std::time::Instant;

use abm::{utility::save_spectrum, DoubleHifiProblemBuilder, HifiProblemBuilder};
use clebsch_gordan::{half_i32, half_integer::HalfI32, half_u32};
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use num::complex::Complex64;
use quantum::{params::{particle::Particle, particle_factory::{self, RotConst}, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{energy_units::{Energy, GHz, Kelvin, MHz}, mass_units::{Dalton, Mass}, Au, Unit}, utility::linspace};
use scattering_problems::{alkali_atoms::AlkaliAtomsProblemBuilder, alkali_rotor_atom::{AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder}, utility::{AnisoHifi, GammaSpinRot, RotorDoubleJTotMax, RotorJMax, RotorLMax}, ScatteringProblem};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::MultiRatioNumerov, propagator::MultiStepRule, single_numerov::SingleRatioNumerov}, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::{Potential, SimplePotential}}, utility::save_data};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "isotropic potential" => |_| Self::potentials(),
    "single channel isotropic scatterings" => |_| Self::single_chan_scatterings(),
    "isotropic feshbach" => |_| Self::feshbach_iso(),
    "rotor levels" => |_| Self::rotor_levels(),
    "rotor potentials" => |_| Self::rotor_potentials(),
    "rotor feshbach" => |_| Self::feshbach_rotor(),
);

impl Problems {
    fn potentials() {
        let particles = get_particles(Energy(1e-7, Kelvin));
        let triplet = triplet_iso(0);
        let singlet = singlet_iso(0);
        let aniso = potential_aniso();

        let distances = linspace(4., 20., 200);
        let triplet_values = distances.iter().map(|&x| triplet.value(x)).collect();
        let singlet_values = distances.iter().map(|&x| singlet.value(x)).collect();
        let aniso_values = distances.iter().map(|&x| aniso.value(x)).collect();

        for config in [0, 1, 2] {
            println!("{config}");
            let triplet = triplet_iso(config);
            let singlet = singlet_iso(config);

            let boundary = Boundary::new(8.5, Direction::Outwards, (1.01, 1.02));
            let mut numerov = SingleRatioNumerov::new(&triplet, &particles, MultiStepRule::default(), boundary);
            numerov.propagate_to(1e4);
            println!("{:.2}", numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re);
    
            let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
            let mut numerov = SingleRatioNumerov::new(&singlet, &particles, MultiStepRule::default(), boundary);
            numerov.propagate_to(1e4);
            println!("{:.2}", numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re);
        }

        let data = vec![distances, triplet_values, singlet_values, aniso_values];
        save_data("CaF_Rb_iso", "distance\ttriplet\tsinglet\taniso", &data)
            .unwrap();
    }

    fn single_chan_scatterings() {
        let particles = get_particles(Energy(1e-7, Kelvin));

        let factors = linspace(0.95, 1.05, 500);

        let scatterings_triplet = factors.iter()
            .map(|x| {
                let mut triplet = Composite::new(Dispersion::new(-3084., -6));
                triplet.add_potential(Dispersion::new(x * 2e9, -12));

                let boundary = Boundary::new(8.5, Direction::Outwards, (1.01, 1.02));
                let mut numerov = SingleRatioNumerov::new(&triplet, &particles, MultiStepRule::default(), boundary);

                numerov.propagate_to(1e4);
                numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re
            })
            .collect();

        let scatterings_singlet = factors.iter()
            .map(|x| {
                let mut singlet = Composite::new(Dispersion::new(-3084., -6));
                singlet.add_potential(Dispersion::new(x * 5e8, -12));

                let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
                let mut numerov = SingleRatioNumerov::new(&singlet, &particles, MultiStepRule::default(), boundary);

                numerov.propagate_to(1e4);
                numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re
            })
            .collect();

        let data = vec![factors, scatterings_triplet, scatterings_singlet];

        save_data("CaF_Rb_1chan_scatterings", "factors\ttriplet\tsinglet", &data)
            .unwrap();
    }

    fn feshbach_iso() {
        ///////////////////////////////////

        let projection = half_i32!(1);
        let entrance = 0;
        let energy = Energy(1e-7, Kelvin);

        let config_triplet = 0;
        let config_singlet = 0;
        
        let mag_fields = linspace(0., 1000., 4000);
        
        linspace(0., 1000., 1000);

        ///////////////////////////////////

        let start = Instant::now();
        
        let scatterings = mag_fields.par_iter().progress().map(|&mag_field| {
            let alkali_problem = get_potential_iso(config_triplet, config_singlet, projection, mag_field, entrance);

            let mut caf_rb = get_particles(energy);
            caf_rb.insert(alkali_problem.asymptotic);
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
            let step_rule = MultiStepRule::default();
            let mut numerov = MultiRatioNumerov::new(potential, &caf_rb, step_rule, boundary);

            numerov.propagate_to(1.5e3);
            numerov.data.calculate_s_matrix().get_scattering_length()
        })
        .collect::<Vec<Complex64>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let scatterings_re = scatterings.iter().map(|x| x.re).collect();
        let scatterings_im = scatterings.iter().map(|x| x.im).collect();

        let header = "mag_field\tscattering_re\tscattering_im";
        let data = vec![mag_fields, scatterings_re, scatterings_im];

        save_data(&format!("CaF_Rb_iso_scatterings_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }

    fn rotor_potentials() {
        ///////////////////////////////////

        let projection = half_i32!(2);
        let channel = 0;

        let config_triplet = 0;
        let config_singlet = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        let distances = linspace(4.2, 30., 200);

        ///////////////////////////////////

        let caf_rb = get_particles(energy_relative);
        let alkali_problem = get_problem(config_triplet, config_singlet, projection, &caf_rb);

        let alkali_problem = alkali_problem.scattering_at_field(100., channel);
        let potential = &alkali_problem.potential;

        let mut mat = Mat::zeros(potential.size(), potential.size());
        let potentials: Vec<Mat<f64>> = distances.iter().map(|&x| {
                potential.value_inplace(x, &mut mat);

                mat.to_owned()
            })
            .collect();

        let header = "distances\tpotentials";
        let mut data = vec![distances];
        for i in 0..potential.size() {
            for j in 0..potential.size() {
                data.push(potentials.iter().map(|p| p[(i, j)]).collect());
            }
        }

        save_data(&format!("CaF_Rb_potentials_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }

    fn rotor_levels() {
        let projection = half_i32!(1);
        let atoms = get_particles(Energy(1e-7, Kelvin));
        let alkali_problem = get_problem(0, 0, projection, &atoms);

        let mag_fields = linspace(0., 2000., 2000);

        let energies: Vec<Vec<f64>> = mag_fields.par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels_at_field(*mag_field);

                levels.iter().map(|x| Energy(*x, Au).to(GHz).value()).collect()
            })
            .collect();
        
        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "caf_rb_levels", &mag_fields, &energies).expect("error while saving abm");
    }

    fn feshbach_rotor() {
        ///////////////////////////////////

        let projection = half_i32!(1);
        let channel = 0;

        let config_triplet = 0;
        let config_singlet = 2;

        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 4000);
        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(config_triplet, config_singlet, projection, &atoms);

        ///////////////////////////////////

        let start = Instant::now();
        let scatterings = mag_fields.par_iter().progress().map(|&mag_field| {
            let mut atoms = get_particles(energy_relative);
            let alkali_problem = alkali_problem.scattering_at_field(mag_field, channel);
            
            atoms.insert(alkali_problem.asymptotic);
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(8.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));
            let step_rule = MultiStepRule::new(1e-3, f64::INFINITY, 500.);
            let mut numerov = MultiRatioNumerov::new(potential, &atoms, step_rule, boundary);

            numerov.propagate_to(1.5e3);
            numerov.data.calculate_s_matrix().get_scattering_length()
        })
        .collect::<Vec<Complex64>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let scatterings_re = scatterings.iter().map(|x| x.re).collect();
        let scatterings_im = scatterings.iter().map(|x| x.im).collect();

        let header = "mag_field\tscattering_re\tscattering_im";
        let data = vec![mag_fields, scatterings_re, scatterings_im];

        save_data(&format!("CaF_Rb_scatterings_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }
}

fn triplet_iso(config: usize) -> Composite<Dispersion> {
    let factors = [1.0286, 0.9717, 1.00268];

    let mut triplet = Composite::new(Dispersion::new(-3084., -6));
    triplet.add_potential(Dispersion::new(factors[config] * 2e9, -12));

    triplet
}

fn singlet_iso(config: usize) -> Composite<Dispersion> {
    let factors = [1.0196, 0.9815, 1.0037];

    let mut singlet = Composite::new(Dispersion::new(-3084., -6));
    singlet.add_potential(Dispersion::new(factors[config] * 5e8, -12));

    singlet
}

fn potential_aniso() -> Composite<Dispersion> {
    let singlet = Composite::new(Dispersion::new(-100., -6));
    singlet
}

fn get_potential_iso(config_triplet: usize, config_singlet: usize, projection: HalfI32, mag_field: f64, entrance: usize) -> ScatteringProblem<impl Potential<Space = Mat<f64>>> {
    let hifi_caf = HifiProblemBuilder::new(half_u32!(1/2), half_u32!(1/2))
        .with_hyperfine_coupling(Energy(120., MHz).to_au());

    let hifi_rb = HifiProblemBuilder::new(half_u32!(1/2), half_u32!(3/2))
        .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

    let triplet = triplet_iso(config_triplet);
    let singlet = singlet_iso(config_singlet);

    AlkaliAtomsProblemBuilder::new(hifi_problem, triplet, singlet)
        .build(mag_field, entrance)
}

fn get_particles(energy: Energy<impl Unit>) -> Particles {
    let caf = Particle::new("CaF", Mass(39.962590850 + 18.998403162, Dalton));
    let rb = particle_factory::create_atom("Rb87").unwrap();

    let mut particles = Particles::new_pair(caf, rb, energy);
    particles.insert(RotorLMax(3));
    particles.insert(RotorJMax(3));
    particles.insert(RotorDoubleJTotMax(3));
    particles.insert(RotConst(Energy(10.3, GHz).to_au()));
    particles.insert(GammaSpinRot(Energy(40., MHz).to_au()));
    particles.insert(AnisoHifi(Energy(14., MHz).to_au()));
    
    particles
}

fn get_problem(config_triplet: usize, config_singlet: usize, projection: HalfI32, particles: &Particles) -> AlkaliRotorAtomProblem<impl SimplePotential + Clone, impl SimplePotential + Clone> {
    let hifi_caf = HifiProblemBuilder::new(half_u32!(1/2), half_u32!(1/2))
        .with_hyperfine_coupling(Energy(120., MHz).to_au());

    let hifi_rb = HifiProblemBuilder::new(half_u32!(1/2), half_u32!(3/2))
        .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

    let triplet = triplet_iso(config_triplet);
    let singlet = singlet_iso(config_singlet);
    let aniso = potential_aniso();

    let triplets = vec![(0, triplet), (2, aniso.clone())];
    let singlets = vec![(0, singlet), (2, aniso)];

    AlkaliRotorAtomProblemBuilder::new(hifi_problem, triplets, singlets)
        .build(particles)
}
