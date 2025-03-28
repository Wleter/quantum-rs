use core::f64;
use std::time::Instant;

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder, utility::save_spectrum};
use clebsch_gordan::{half_integer::HalfI32, hi32, hu32};
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use num::complex::Complex64;
use quantum::{
    params::{
        Params,
        particle::Particle,
        particle_factory::{self, RotConst},
        particles::Particles,
    },
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::{
        Au,
        energy_units::{Energy, EnergyUnit, GHz, Kelvin, MHz},
        mass_units::{Dalton, Mass},
    },
    utility::linspace,
};
use scattering_problems::{
    FieldScatteringProblem, IndexBasisDescription, ScatteringProblem,
    alkali_atoms::AlkaliAtomsProblemBuilder,
    alkali_rotor_atom::{
        AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder, TramBasisRecipe, TramStates,
        UncoupledRotorBasisRecipe,
    },
    uncoupled_alkali_rotor_atom::UncoupledAlkaliRotorAtomStates,
    utility::{AnisoHifi, GammaSpinRot},
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    numerovs::{
        LocalWavelengthStepRule, multi_numerov::MultiRNumerov, single_numerov::SingleRNumerov,
    },
    potentials::{
        composite_potential::Composite,
        dispersion_potential::Dispersion,
        potential::{MatPotential, Potential, SimplePotential},
    },
    propagator::{CoupledEquation, Propagator, SingleEquation},
    utility::save_data,
};

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
    "N_max convergence" => |_| Self::n_max_convergence(),
    "N_max convergence uncoupled representation" => |_| Self::n_max_convergence_uncoupled(),
    "rotor feshbach uncoupled" => |_| Self::uncoupled_feshbach_rotor(),
);

impl Problems {
    const POTENTIAL_CONFIGS: [(usize, usize); 4] = [(0, 0), (1, 0), (1, 1), (0, 2)];

    fn potentials() {
        let particles = get_particles(Energy(1e-7, Kelvin), hi32!(1));
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
            let eq = SingleEquation::from_particles(&triplet, &particles);
            let step_rule = LocalWavelengthStepRule::default();
            let mut numerov = SingleRNumerov::new(eq, boundary, step_rule);

            numerov.propagate_to(1e4);
            println!("{:.2}", numerov.s_matrix().get_scattering_length().re);

            let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
            let eq = SingleEquation::from_particles(&singlet, &particles);
            let step_rule = LocalWavelengthStepRule::default();

            let mut numerov = SingleRNumerov::new(eq, boundary, step_rule);
            numerov.propagate_to(1e4);

            println!("{:.2}", numerov.s_matrix().get_scattering_length().re);
        }

        let data = vec![distances, triplet_values, singlet_values, aniso_values];
        save_data("CaF_Rb_iso", "distance\ttriplet\tsinglet\taniso", &data).unwrap();
    }

    fn single_chan_scatterings() {
        let particles = get_particles(Energy(1e-7, Kelvin), hi32!(1));

        let factors = linspace(0.95, 1.05, 500);

        let scatterings_triplet = factors
            .iter()
            .map(|x| {
                let mut triplet = Composite::new(Dispersion::new(-3084., -6));
                triplet.add_potential(Dispersion::new(x * 2e9, -12));

                let boundary = Boundary::new(8.5, Direction::Outwards, (1.01, 1.02));
                let eq = SingleEquation::from_particles(&triplet, &particles);
                let step_rule = LocalWavelengthStepRule::default();
                let mut numerov = SingleRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(1e4);
                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let scatterings_singlet = factors
            .iter()
            .map(|x| {
                let mut singlet = Composite::new(Dispersion::new(-3084., -6));
                singlet.add_potential(Dispersion::new(x * 5e8, -12));

                let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
                let eq = SingleEquation::from_particles(&singlet, &particles);
                let step_rule = LocalWavelengthStepRule::default();
                let mut numerov = SingleRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(1e4);
                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let data = vec![factors, scatterings_triplet, scatterings_singlet];

        save_data(
            "CaF_Rb_1chan_scatterings",
            "factors\ttriplet\tsinglet",
            &data,
        )
        .unwrap();
    }

    fn feshbach_iso() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let mag_fields = linspace(0., 1000., 4000);
            let projection = hi32!(1);
            let energy = Energy(1e-7, Kelvin);

            ///////////////////////////////////

            let atoms = get_particles(energy, projection);

            let start = Instant::now();
            let scatterings = mag_fields
                .par_iter()
                .progress()
                .map_with(atoms, |atoms, &mag_field| {
                    let alkali_problem =
                        get_potential_iso(config_triplet, config_singlet, projection, mag_field);

                    atoms.insert(alkali_problem.asymptotic);
                    let potential = &alkali_problem.potential;

                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary =
                        Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                    let eq = CoupledEquation::from_particles(potential, &atoms);
                    let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                    numerov.propagate_to(1.5e3);
                    numerov.s_matrix().get_scattering_length()
                })
                .collect::<Vec<Complex64>>();

            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());

            let scatterings_re = scatterings.iter().map(|x| x.re).collect();
            let scatterings_im = scatterings.iter().map(|x| x.im).collect();

            let header = "mag_field\tscattering_re\tscattering_im";
            let data = vec![mag_fields, scatterings_re, scatterings_im];

            save_data(
                &format!("CaF_Rb_iso_scatterings_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
    }

    fn rotor_levels() {
        let projection = hi32!(1);
        let basis_recipe = TramBasisRecipe {
            l_max: 5,
            n_max: 5,
            ..Default::default()
        };

        let basis_recipe_uncoupled = UncoupledRotorBasisRecipe {
            l_max: 5,
            n_max: 5,
            ..Default::default()
        };

        let atoms = get_particles(Energy(1e-7, Kelvin), projection);
        let alkali_problem = get_problem(0, 0, &atoms, &basis_recipe);

        let mag_fields = linspace(0., 200., 200);

        let energies: Vec<Vec<f64>> = mag_fields
            .par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels(*mag_field, Some(0));

                levels
                    .iter()
                    .map(|x| Energy(*x, Au).to(GHz).value())
                    .collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "caf_rb_levels", &mag_fields, &energies)
            .expect("error while saving abm");

        let atoms = get_particles(Energy(1e-7, Kelvin), projection);
        let alkali_problem = get_problem_uncoupled(0, 0, &atoms, &basis_recipe_uncoupled);

        let energies: Vec<Vec<f64>> = mag_fields
            .par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels(*mag_field, Some(0));

                levels
                    .iter()
                    .map(|x| Energy(*x, Au).to(GHz).value())
                    .collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, "caf_rb_levels_uncoupled", &mag_fields, &energies)
            .expect("error while saving abm");
    }

    fn rotor_potentials() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let projection = hi32!(2);
            let energy_relative = Energy(1e-7, Kelvin);
            let distances = linspace(4.2, 30., 200);
            let basis_recipe = TramBasisRecipe {
                n_max: 5,
                l_max: 5,
                ..Default::default()
            };

            ///////////////////////////////////

            let caf_rb = get_particles(energy_relative, projection);
            let alkali_problem =
                get_problem(config_triplet, config_singlet, &caf_rb, &basis_recipe);

            let alkali_problem = alkali_problem.scattering_for(100.);
            let potential = &alkali_problem.potential;

            let mut mat = Mat::zeros(potential.size(), potential.size());
            let potentials: Vec<Mat<f64>> = distances
                .iter()
                .map(|&x| {
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

            save_data(
                &format!("CaF_Rb_potentials_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
    }

    fn feshbach_rotor() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let projection = hi32!(1);
            let energy_relative = Energy(1e-7, Kelvin);
            let mag_fields = linspace(0., 1000., 4000);
            let basis_recipe = TramBasisRecipe {
                n_max: 5,
                l_max: 5,
                ..Default::default()
            };

            let atoms = get_particles(energy_relative, projection);
            let alkali_problem = get_problem(config_triplet, config_singlet, &atoms, &basis_recipe);

            ///////////////////////////////////

            let start = Instant::now();
            let scatterings = mag_fields
                .par_iter()
                .progress()
                .map_with(atoms, |atoms, &mag_field| {
                    let alkali_problem = alkali_problem.scattering_for(mag_field);

                    atoms.insert(alkali_problem.asymptotic);
                    let potential = &alkali_problem.potential;

                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary =
                        Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                    let eq = CoupledEquation::from_particles(potential, &atoms);
                    let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                    numerov.propagate_to(1.5e3);
                    numerov.s_matrix().get_scattering_length()
                })
                .collect::<Vec<Complex64>>();

            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());

            let scatterings_re = scatterings.iter().map(|x| x.re).collect();
            let scatterings_im = scatterings.iter().map(|x| x.im).collect();

            let header = "mag_field\tscattering_re\tscattering_im";
            let data = vec![mag_fields, scatterings_re, scatterings_im];

            save_data(
                &format!("CaF_Rb_scatterings_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
    }

    fn n_max_convergence() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let projection = hi32!(1);
            let n_maxes: Vec<u32> = (0..20).collect();

            let energy_relative = Energy(1e-7, Kelvin);
            let mag_field = 500.;

            let atoms = get_particles(energy_relative, projection);

            ///////////////////////////////////

            let start = Instant::now();
            let scatterings = n_maxes
                .par_iter()
                .progress()
                .map_with(atoms, |atoms, &n_max| {
                    let basis_recipe = TramBasisRecipe {
                        n_max: n_max,
                        l_max: n_max,
                        ..Default::default()
                    };

                    let alkali_problem =
                        get_problem(config_triplet, config_singlet, &atoms, &basis_recipe);
                    let alkali_problem = alkali_problem.scattering_for(mag_field);
                    atoms.insert(alkali_problem.asymptotic);
                    let potential = &alkali_problem.potential;

                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary =
                        Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                    let eq = CoupledEquation::from_particles(potential, &atoms);
                    let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                    numerov.propagate_to(1.5e3);
                    numerov.s_matrix().get_scattering_length()
                })
                .collect::<Vec<Complex64>>();

            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());

            let n_maxes = n_maxes.iter().map(|&x| x as f64).collect();
            let scatterings_re = scatterings.iter().map(|x| x.re).collect();
            let scatterings_im = scatterings.iter().map(|x| x.im).collect();

            let header = "mag_field\tscattering_re\tscattering_im";
            let data = vec![n_maxes, scatterings_re, scatterings_im];

            save_data(
                &format!("CaF_Rb_n_max_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
    }

    fn n_max_convergence_uncoupled() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let projection = hi32!(1);
            let n_maxes: Vec<u32> = (0..=3).collect();

            let energy_relative = Energy(1e-7, Kelvin);
            let mag_field = 500.;

            let atoms = get_particles(energy_relative, projection);

            ///////////////////////////////////

            let start = Instant::now();
            let scatterings = n_maxes
                .par_iter()
                .progress()
                .map_with(atoms, |atoms, &n_max| {
                    let basis_recipe = UncoupledRotorBasisRecipe {
                        n_max: n_max,
                        l_max: n_max,
                        ..Default::default()
                    };

                    let alkali_problem = get_problem_uncoupled(
                        config_triplet,
                        config_singlet,
                        &atoms,
                        &basis_recipe,
                    );
                    let alkali_problem = alkali_problem.scattering_for(mag_field);
                    atoms.insert(alkali_problem.asymptotic.clone());
                    let potential = &alkali_problem.potential;

                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary =
                        Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                    let eq = CoupledEquation::from_particles(potential, &atoms);
                    let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                    numerov.propagate_to(1.5e3);
                    numerov.s_matrix().get_scattering_length()
                })
                .collect::<Vec<Complex64>>();

            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());

            let n_maxes = n_maxes.iter().map(|&x| x as f64).collect();
            let scatterings_re = scatterings.iter().map(|x| x.re).collect();
            let scatterings_im = scatterings.iter().map(|x| x.im).collect();

            let header = "mag_field\tscattering_re\tscattering_im";
            let data = vec![n_maxes, scatterings_re, scatterings_im];

            save_data(
                &format!("CaF_Rb_n_max_uncoupled_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
    }

    fn uncoupled_feshbach_rotor() {
        for (config_triplet, config_singlet) in Self::POTENTIAL_CONFIGS {
            let projection = hi32!(1);

            let energy_relative = Energy(1e-7, Kelvin);
            let mag_fields = linspace(0., 1000., 4000);

            let basis_recipe = UncoupledRotorBasisRecipe {
                n_max: 2,
                l_max: 2,
                ..Default::default()
            };

            let atoms = get_particles(energy_relative, projection);
            let alkali_problem =
                get_problem_uncoupled(config_triplet, config_singlet, &atoms, &basis_recipe);

            ///////////////////////////////////

            let start = Instant::now();
            let scatterings = mag_fields
                .par_iter()
                .progress()
                .map_with(atoms, |atoms, &mag_field| {
                    let alkali_problem = alkali_problem.scattering_for(mag_field);

                    atoms.insert(alkali_problem.asymptotic);
                    let potential = &alkali_problem.potential;

                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary =
                        Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                    let eq = CoupledEquation::from_particles(potential, &atoms);
                    let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                    numerov.propagate_to(1.5e3);
                    numerov.s_matrix().get_scattering_length()
                })
                .collect::<Vec<Complex64>>();

            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());

            let scatterings_re = scatterings.iter().map(|x| x.re).collect();
            let scatterings_im = scatterings.iter().map(|x| x.im).collect();

            let header = "mag_field\tscattering_re\tscattering_im";
            let data = vec![mag_fields, scatterings_re, scatterings_im];

            save_data(
                &format!("CaF_Rb_uncoupled_scatterings_{config_triplet}_{config_singlet}"),
                header,
                &data,
            )
            .unwrap()
        }
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

fn get_potential_iso(
    config_triplet: usize,
    config_singlet: usize,
    projection: HalfI32,
    mag_field: f64,
) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
    let hifi_caf = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
        .with_hyperfine_coupling(Energy(120., MHz).to_au());

    let hifi_rb = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(3 / 2))
        .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

    let triplet = triplet_iso(config_triplet);
    let singlet = singlet_iso(config_singlet);

    AlkaliAtomsProblemBuilder::new(hifi_problem, triplet, singlet).build(mag_field)
}

fn get_particles(energy: Energy<impl EnergyUnit>, projection: HalfI32) -> Particles {
    let caf = Particle::new("CaF", Mass(39.962590850 + 18.998403162, Dalton));
    let rb = particle_factory::create_atom("Rb87").unwrap();

    let mut particles = Particles::new_pair(caf, rb, energy);
    particles.insert(RotConst(Energy(10.3, GHz).to_au()));
    particles.insert(GammaSpinRot(Energy(40., MHz).to_au()));
    particles.insert(AnisoHifi(Energy(3. * 14., MHz).to_au()));

    let hifi_caf = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
        .with_hyperfine_coupling(Energy(120., MHz).to_au());
    let hifi_rb = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(3 / 2))
        .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

    particles.insert(hifi_problem);

    particles
}

fn get_problem(
    config_triplet: usize,
    config_singlet: usize,
    params: &Params,
    basis_recipe: &TramBasisRecipe,
) -> AlkaliRotorAtomProblem<
    TramStates,
    impl SimplePotential + Clone + use<>,
    impl SimplePotential + Clone + use<>,
> {
    let triplet = triplet_iso(config_triplet);
    let singlet = singlet_iso(config_singlet);
    let aniso = potential_aniso();

    let triplets = vec![(0, triplet), (2, aniso.clone())];
    let singlets = vec![(0, singlet), (2, aniso)];

    AlkaliRotorAtomProblemBuilder::new(triplets, singlets).build(params, basis_recipe)
}

fn get_problem_uncoupled(
    config_triplet: usize,
    config_singlet: usize,
    params: &Params,
    basis_recipe: &UncoupledRotorBasisRecipe,
) -> AlkaliRotorAtomProblem<
    UncoupledAlkaliRotorAtomStates,
    impl SimplePotential + Clone + use<>,
    impl SimplePotential + Clone + use<>,
> {
    let triplet = triplet_iso(config_triplet);
    let singlet = singlet_iso(config_singlet);
    let aniso = potential_aniso();

    let triplets = vec![(0, triplet), (2, aniso.clone())];
    let singlets = vec![(0, singlet), (2, aniso)];

    AlkaliRotorAtomProblemBuilder::new(triplets, singlets).build_uncoupled(params, basis_recipe)
}
