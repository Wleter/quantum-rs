use std::{
    f64::consts::PI,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder};
use clebsch_gordan::{half_integer::HalfI32, hi32, hu32};
use faer::Mat;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    params::{
        particle::Particle, particle_factory::{create_atom, RotConst}, particles::Particles, Params
    },
    problem_selector::{get_args, ProblemSelector},
    problems_impl,
    units::{
        distance_units::{Angstrom, Distance}, energy_units::{CmInv, Energy, EnergyUnit, GHz, Kelvin}, mass_units::{Dalton, Mass}, Au, MHz, Unit
    },
    utility::{legendre_polynomials, linspace},
};
use scattering_problems::{
    FieldScatteringProblem,
    alkali_atoms::AlkaliAtomsProblemBuilder,
    alkali_rotor_atom::{
        AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder, TramBasisRecipe, TramStates,
    },
    potential_interpolation::{PotentialArray, TransitionedPotential, interpolate_potentials},
    rkhs_interpolation::RKHSInterpolation,
    rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder},
    utility::{AnisoHifi, GammaSpinRot},
};
use scattering_solver::{
    boundary::{Boundary, Direction}, log_derivatives::johnson::Johnson, numerovs::{
        multi_numerov::MultiRNumerov, propagator_watcher::PropagatorLogging, LocalWavelengthStepRule
    }, observables::{bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence}, s_matrix::{ScatteringDependence, ScatteringObservables}}, potentials::{
        composite_potential::Composite, dispersion_potential::Dispersion, multi_coupling::MultiCoupling, pair_potential::PairPotential, potential::{Potential, ScaledPotential, SimplePotential}
    }, propagator::{CoupledEquation, Propagator}, utility::{save_data, save_serialize}
};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "potentials" => |_| Self::potentials(),
    "single cross section calculation" => |_| Self::single_cross_sections(),
    "atom approximation cross section calculation" => |_| Self::atom_approximation_cross_sections(),
    "cross sections calculation" => |_| Self::cross_sections(),
    "a_length potential scaling" => |_| Self::potential_scaling_propagation(),
    "spinless convergence" => |_| Self::spinless_convergence(),
    "potential scaled scattering calculation" => |_| Self::scattering_scaled(),
    "bound states" => |_| Self::bound_states(),
    "bound states potential scaling singlet" => |_| Self::bounds_singlet_scaling(),
    "bound states potential scaling triplet" => |_| Self::bounds_triplet_scaling(),
);

impl Problems {
    fn potentials() {
        let distances = linspace(5., 80., 800);

        let [pot_array_singlet, pot_array_triplet] = read_potentials(25);
        let mut data = vec![pot_array_triplet.distances.clone()];
        for (_, p) in &pot_array_triplet.potentials {
            data.push(p.clone());
        }

        save_data(
            "SrF_Rb_triplet_dec",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();

        let mut data = vec![pot_array_singlet.distances.clone()];
        for (_, p) in &pot_array_singlet.potentials {
            data.push(p.clone());
        }

        save_data(
            "SrF_Rb_singlet_dec",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();

        let [pot_array_singlet, pot_array_triplet] = read_extended(25);
        let interpolated = get_interpolated(&pot_array_triplet);
        let mut data = vec![distances.clone()];
        for (_, p) in &interpolated {
            let values = distances.iter().map(|&x| p.value(x)).collect();

            data.push(values);
        }

        save_data(
            "SrF_Rb_triplet_dec_interpolated",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();

        let interpolated = get_interpolated(&pot_array_singlet);
        let mut data = vec![distances.clone()];
        for (_, p) in &interpolated {
            let values = distances.iter().map(|&x| p.value(x)).collect();
            data.push(values);
        }

        save_data(
            "SrF_Rb_singlet_dec_interpolated",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();
    }

    fn single_cross_sections() {
        let entrance = 0;
        let mag_field = 100.0;

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);

        let basis_recipe = TramBasisRecipe {
            l_max: 40,
            n_max: 40,
            n_tot_max: 0,
            ..Default::default()
        };

        ///////////////////////////////////

        let mut atoms = get_particles(energy_relative, projection);
        let alkali_problem = get_problem(&atoms, &basis_recipe);

        let alkali_problem = alkali_problem.scattering_for(mag_field);
        let mut asymptotic = alkali_problem.asymptotic;
        asymptotic.entrance = entrance;
        atoms.insert(asymptotic);
        let potential = &alkali_problem.potential;

        let boundary = Boundary::new_multi_vanishing(5.0, Direction::Outwards, potential.size());
        let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
        let eq = CoupledEquation::from_particles(potential, &atoms);
        let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

        numerov.propagate_to_with(1500., &mut PropagatorLogging::default());

        let scattering = numerov.s_matrix().observables();

        println!("{:?}", scattering);
    }

    fn atom_approximation_cross_sections() {
        let entrance = 4;
        let mag_fields = linspace(0., 2000., 1000);

        let projection = hi32!(-1);
        let energy_relative = Energy(1e-7, Kelvin);

        ////////////////////////////////////

        let [singlet, triplet] = read_extended(25);

        let singlets = get_interpolated(&singlet);
        let triplets = get_interpolated(&triplet);
        let singlet = singlets[0].1.clone();
        let triplet = triplets[0].1.clone();

        let atoms = get_particles(energy_relative, projection);
        let hifi_problem = atoms.get::<DoubleHifiProblemBuilder>().unwrap();

        let alkali_problem =
            AlkaliAtomsProblemBuilder::new(hifi_problem.to_owned(), singlet, triplet);

        ///////////////////////////////////

        let start = Instant::now();
        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &mag_field| {
                let alkali_problem = alkali_problem.clone().build(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let boundary =
                    Boundary::new_multi_vanishing(5.0, Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = ScatteringDependence {
            parameters: mag_fields,
            observables: scatterings,
        };

        save_serialize("SrF_Rb_scatterings_n_0", &data).unwrap()
    }

    fn cross_sections() {
        let entrance = 0;

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 500);
        let basis_recipe = TramBasisRecipe {
            l_max: 10,
            n_max: 10,
            n_tot_max: 0,
            ..Default::default()
        };

        let atoms = get_particles(energy_relative, projection);
        let alkali_problem = get_problem(&atoms, &basis_recipe);

        ///////////////////////////////////

        let start = Instant::now();
        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = ScatteringDependence {
            parameters: mag_fields,
            observables: scatterings,
        };

        let filename = format!(
            "SrF_Rb_scatterings_ground_n_max_{}_n_tot_max_{}",
            basis_recipe.n_max, basis_recipe.n_tot_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn potential_scaling_propagation() {
        let scalings = linspace(1.0, 1.05, 500);
        let energy_relative = Energy(1e-7, Kelvin);
        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 5,
            n_max: 5,
            ..Default::default()
        };

        let [singlet, triplet] = read_extended(25);
        let singlets = get_interpolated(&singlet);
        let triplets = get_interpolated(&triplet);

        let atoms = get_particles(energy_relative, hi32!(0));

        let start = Instant::now();
        let singlet_scattering = scalings
            .par_iter()
            .progress()
            .map_with(atoms.clone(), |atoms, &scaling| {
                let potential = singlets
                    .iter()
                    .map(|(lambda, p)| {
                        (
                            *lambda,
                            ScaledPotential {
                                potential: p.clone(),
                                scaling,
                            },
                        )
                    })
                    .collect();

                let problem = RotorAtomProblemBuilder::new(potential).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
                numerov.propagate_to(1500.);

                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let start = Instant::now();
        let triplet_scattering = scalings
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &scaling| {
                let potential = triplets
                    .iter()
                    .map(|(lambda, p)| {
                        (
                            *lambda,
                            ScaledPotential {
                                potential: p.clone(),
                                scaling,
                            },
                        )
                    })
                    .collect();

                let problem = RotorAtomProblemBuilder::new(potential).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
                numerov.propagate_to(1500.);

                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = vec![scalings, singlet_scattering, triplet_scattering];

        save_data(
            &format!("srf_rb_potential_scaling_n_max_{}", basis_recipe.n_max),
            "scaling\tsinglet\ttriplet",
            &data,
        )
        .unwrap();
    }

    fn spinless_convergence() {
        let n_maxes: Vec<u32> = (0..200).collect();
        let energy_relative = Energy(1e-7, Kelvin);

        let [singlet, triplet] = read_extended(25);
        let singlets = get_interpolated(&singlet);
        let triplets = get_interpolated(&triplet);

        let atoms = get_particles(energy_relative, hi32!(0));

        let start = Instant::now();
        let singlet_scattering = n_maxes
            .par_iter()
            .progress()
            .map_with(atoms.clone(), |atoms, &n_max| {
                let basis_recipe = RotorAtomBasisRecipe {
                    l_max: n_max,
                    n_max: n_max,
                    ..Default::default()
                };

                let problem =
                    RotorAtomProblemBuilder::new(singlets.clone()).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
                numerov.propagate_to(1500.);

                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());
        let start = Instant::now();

        let triplet_scattering = n_maxes
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &n_max| {
                let basis_recipe = RotorAtomBasisRecipe {
                    l_max: n_max,
                    n_max: n_max,
                    ..Default::default()
                };

                let problem =
                    RotorAtomProblemBuilder::new(triplets.clone()).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
                numerov.propagate_to(1500.);

                numerov.s_matrix().get_scattering_length().re
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let n_maxes = n_maxes.into_iter().map(|x| x as f64).collect();
        let data = vec![n_maxes, singlet_scattering, triplet_scattering];

        save_data(
            &format!("srf_rb_potential_scattering_convergence"),
            "scaling\tsinglet\ttriplet",
            &data,
        )
        .unwrap();
    }

    fn scattering_scaled() {
        let n_max = 5;
        let singlet_scaling = 1.0174797334102; // #1 a_0 = -50 Angstrom,
        let triplet_scaling = 0.9587806804328; // #1 a_0 = 0,

        // let n_max = 5;
        // let singlet_scaling = 1.0349451752971;   // #2 a_0 = -50 Angstrom,
        // let triplet_scaling = 1.0657638160382;   // #2 a_0 = 0,

        // let n_max = 0;
        // let singlet_scaling = 1.0056292421443;
        // let triplet_scaling = 1.0234326602081;

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 2000., 1000);
        let basis_recipe = TramBasisRecipe {
            l_max: n_max,
            n_max: n_max,
            ..Default::default()
        };

        let atoms = get_particles(energy_relative, projection);

        let [singlet, triplet] = read_extended(25);
        let singlets = get_interpolated(&singlet);
        let triplets = get_interpolated(&triplet);

        let triplets = triplets
            .into_iter()
            .map(|(lambda, p)| {
                (
                    lambda,
                    ScaledPotential {
                        potential: p,
                        scaling: triplet_scaling,
                    },
                )
            })
            .collect();

        let singlets = singlets
            .into_iter()
            .map(|(lambda, p)| {
                (
                    lambda,
                    ScaledPotential {
                        potential: p,
                        scaling: singlet_scaling,
                    },
                )
            })
            .collect();

        let alkali_problem =
            AlkaliRotorAtomProblemBuilder::new(triplets, singlets).build(&atoms, &basis_recipe);

        /////////////////////////////////////////////////

        let start = Instant::now();
        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &mag_field| {
                let alkali_problem = alkali_problem.scattering_for(mag_field);
                atoms.insert(alkali_problem.asymptotic);
                let potential = &alkali_problem.potential;

                let id = Mat::<f64>::identity(potential.size(), potential.size());
                let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = ScatteringDependence {
            parameters: mag_fields,
            observables: scatterings,
        };

        save_serialize(&format!("SrF_Rb_scaled_scattering_n_{n_max}_v4"), &data).unwrap()
    }

    fn bound_states() {
        let entrance = 0;

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 500);
        let basis_recipe = TramBasisRecipe {
            l_max: 10,
            n_max: 10,
            n_tot_max: 0,
            ..Default::default()
        };

        let atoms = get_particles(energy_relative, projection);
        let alkali_problem = get_problem(&atoms, &basis_recipe);

        let energy_range = (Energy(-1., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let bound_states = mag_fields
            .iter()
            .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem.bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect::<Vec<BoundStates>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: mag_fields,
            bound_states,
        };

        let filename = format!(
            "SrF_Rb_bound_states_n_max_{}_n_tot_max_{}",
            basis_recipe.n_max, basis_recipe.n_tot_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn bounds_singlet_scaling() {
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, hi32!(0));

        let [singlet, _] = read_extended(25);
        let singlets = get_interpolated(&singlet);
        
        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings = linspace(0., 1e7, 100);

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let singlet_bounds = scalings
            .par_iter()
            .progress()
            .map_with(atoms.clone(), |atoms, &scaling| {
                let singlets = singlets.clone();
                // let singlets = singlets 
                    // .iter()
                    // .map(|(lambda, p)| {
                    //     (
                    //         *lambda,
                    //         ScaledPotential {
                    //             potential: p.clone(),
                    //             scaling,
                    //         },
                    //     )
                    // })
                    // .collect();

                let problem = RotorAtomProblemBuilder::new(singlets)
                    .build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = problem.potential;

                let dispersion = Dispersion::new(scaling, -12);
                let potential_shift = MultiCoupling::new(potential.size(), vec![(dispersion, 2, 2)]);

                let potential = PairPotential::new(potential, potential_shift);
    
                let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem.bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: singlet_bounds,
        }; 
        let filename = format!("SrF_Rb_singlet_bound_states_n_max_{}_shift_2", basis_recipe.n_max);

        save_serialize(&filename, &data).unwrap()
    }

    fn bounds_triplet_scaling() {
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, hi32!(0));

        let [_, triplet] = read_extended(25);
        let triplets = get_interpolated(&triplet);
        
        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings = linspace(1., 1.1, 500);

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let triplet_bounds = scalings
            .par_iter()
            .progress()
            .map_with(atoms.clone(), |atoms, &scaling| {
                let triplets = triplets
                    .iter()
                    .map(|(lambda, p)| {
                        (
                            *lambda,
                            ScaledPotential {
                                potential: p.clone(),
                                scaling,
                            },
                        )
                    })
                    .collect();

                let problem = RotorAtomProblemBuilder::new(triplets)
                    .build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &problem.potential;
    
                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem.bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: triplet_bounds,
        };

        let filename = format!("SrF_Rb_triplet_bound_states_n_max_{}", basis_recipe.n_max);

        save_serialize(&filename, &data).unwrap()
    }
}

fn get_particles(energy: Energy<impl EnergyUnit>, projection: HalfI32) -> Particles {
    let rb = create_atom("Rb87").unwrap();
    let srf = Particle::new("SrF", Mass(88. + 19., Dalton));

    let mut particles = Particles::new_pair(rb, srf, energy);

    let mass = 47.9376046914861;
    particles.insert(Mass(mass, Dalton).to(Au));

    particles.insert(RotConst(Energy(0.24975935, CmInv).to_au()));
    particles.insert(GammaSpinRot(Energy(2.4974e-3, CmInv).to_au()));
    particles.insert(AnisoHifi(Energy(1.0096e-3, CmInv).to_au()));

    let hifi_srf = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
        .with_hyperfine_coupling(Energy(3.2383e-3 + 1.0096e-3 / 3., CmInv).to_au());

    let hifi_rb = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(3 / 2))
        .with_hyperfine_coupling(Energy(6.834682610904290 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_srf, hifi_rb).with_projection(projection);
    particles.insert(hifi_problem);

    particles
}

fn get_problem(
    params: &Params,
    basis_recipe: &TramBasisRecipe,
) -> AlkaliRotorAtomProblem<
    TramStates,
    impl SimplePotential + Clone + use<>,
    impl SimplePotential + Clone + use<>,
> {
    let [singlet, triplet] = read_extended(25);

    let singlets = get_interpolated(&singlet);
    let triplets = get_interpolated(&triplet);

    AlkaliRotorAtomProblemBuilder::new(triplets, singlets).build(params, basis_recipe)
}

fn read_potentials(max_degree: u32) -> [PotentialArray; 2] {
    let filename = "Rb_SrF/pot.data.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f = BufReader::new(f);

    let filename = "Rb_SrF/casscf_ex.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f2 = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f2 = BufReader::new(f2);

    let angle_count = 1 + 180 / 5;
    let r_count = 30;
    let mut values_singlet = Mat::zeros(angle_count, r_count);
    let mut values_triplet = Mat::zeros(angle_count, r_count);
    let mut distances = vec![0.; r_count];
    let angles: Vec<f64> = (0..=180).step_by(5).map(|x| x as f64 / 180. * PI).collect();

    for ((i, line_triplet), line_diff) in f.lines().skip(1).enumerate().zip(f2.lines().skip(1)) {
        let line_triplet = line_triplet.unwrap();
        let splitted_triplet: Vec<&str> = line_triplet.trim().split_whitespace().collect();

        let r: f64 = splitted_triplet[0].parse().unwrap();
        let value: f64 = splitted_triplet[1].parse().unwrap();

        let angle_index = i / r_count;
        let r_index = i % r_count;

        if angle_index > 0 {
            assert!(distances[r_index] == Distance(r, Angstrom).to_au())
        }

        let line_diff = line_diff.unwrap();
        let splitted_diff: Vec<&str> = line_diff.trim().split_whitespace().collect();

        let r_diff: f64 = splitted_diff[0].parse().unwrap();
        let value_diff: f64 = splitted_diff[1].parse().unwrap();

        assert!(r_diff == r);

        distances[r_index] = Distance(r, Angstrom).to_au();
        values_singlet[(angle_index, r_index)] = Energy(value - value_diff, CmInv).to_au();
        values_triplet[(angle_index, r_index)] = Energy(value, CmInv).to_au();
    }

    let filename = "weights.txt";
    path.pop();
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let mut f = BufReader::new(f);

    let mut weights = String::new();
    f.read_line(&mut weights).unwrap();
    let weights: Vec<f64> = weights
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    let mut potentials_singlet = Vec::new();
    let mut potentials_triplet = Vec::new();
    let polynomials: Vec<Vec<f64>> = angles
        .iter()
        .map(|x| legendre_polynomials(max_degree, x.cos()))
        .collect();

    for lambda in 0..=max_degree {
        let mut lambda_values = Vec::new();
        for values_col in values_triplet.col_iter() {
            let value: f64 = weights
                .iter()
                .zip(values_col.iter())
                .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                .map(|((w, v), p)| (lambda as f64 + 0.5) * w * v * p)
                .sum();
            lambda_values.push(value)
        }
        potentials_triplet.push((lambda as u32, lambda_values));

        let mut lambda_values = Vec::new();
        for values_col in values_singlet.col_iter() {
            let value: f64 = weights
                .iter()
                .zip(values_col.iter())
                .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                .map(|((w, v), p)| (lambda as f64 + 0.5) * w * v * p)
                .sum();
            lambda_values.push(value)
        }
        potentials_singlet.push((lambda as u32, lambda_values));
    }

    let singlets = PotentialArray {
        potentials: potentials_singlet,
        distances: distances.clone(),
    };

    let triplets = PotentialArray {
        potentials: potentials_triplet,
        distances,
    };

    [singlets, triplets]
}

fn read_extended(max_degree: u32) -> [PotentialArray; 2] {
    let filename = "Rb_SrF/pot.data.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f = BufReader::new(f);

    let filename = "Rb_SrF/casscf_ex.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f2 = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f2 = BufReader::new(f2);

    let angle_count = 1 + 180 / 5;
    let r_count = 30;
    let mut values_triplet = Mat::zeros(r_count, angle_count);
    let mut values_exch = Mat::zeros(r_count, angle_count);
    let mut distances = vec![0.; r_count];
    let angles: Vec<f64> = (0..=180).step_by(5).map(|x| x as f64 / 180. * PI).collect();

    for ((i, line_triplet), line_diff) in f.lines().skip(1).enumerate().zip(f2.lines().skip(1)) {
        let line_triplet = line_triplet.unwrap();
        let splitted_triplet: Vec<&str> = line_triplet.trim().split_whitespace().collect();

        let r: f64 = splitted_triplet[0].parse().unwrap();
        let value: f64 = splitted_triplet[1].parse().unwrap();

        let angle_index = i / r_count;
        let r_index = i % r_count;

        if angle_index > 0 {
            assert!(distances[r_index] == Distance(r, Angstrom).to_au())
        }

        let line_diff = line_diff.unwrap();
        let splitted_diff: Vec<&str> = line_diff.trim().split_whitespace().collect();

        let r_diff: f64 = splitted_diff[0].parse().unwrap();
        let value_diff: f64 = splitted_diff[1].parse().unwrap();

        assert!(r_diff == r);

        distances[r_index] = Distance(r, Angstrom).to_au();
        values_triplet[(r_index, angle_index)] = Energy(value, CmInv).to_au();
        values_exch[(r_index, angle_index)] = Energy(value_diff, CmInv).to_au();
    }

    let rkhs_triplet = values_triplet
        .col_iter()
        .map(|col| RKHSInterpolation::new(&distances, &col.iter().copied().collect::<Vec<f64>>()))
        .collect::<Vec<RKHSInterpolation>>();

    let rkhs_exch = values_exch
        .col_iter()
        .map(|col| RKHSInterpolation::new(&distances, &col.iter().copied().collect::<Vec<f64>>()))
        .collect::<Vec<RKHSInterpolation>>();

    let filename = "weights.txt";
    path.pop();
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let mut f = BufReader::new(f);

    let mut weights = String::new();
    f.read_line(&mut weights).unwrap();
    let weights: Vec<f64> = weights
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    let mut tail_far = Vec::new();
    for _ in 0..=max_degree {
        tail_far.push(Composite::new(Dispersion::new(0., 0)));
    }

    tail_far[0]
        .add_potential(Dispersion::new(-3495.30040855597, -6))
        .add_potential(Dispersion::new(-516911.950541056, -8));
    tail_far[1]
        .add_potential(Dispersion::new(17274.8363457991, -7))
        .add_potential(Dispersion::new(4768422.32042577, -9));
    tail_far[2]
        .add_potential(Dispersion::new(-288.339392609436, -6))
        .add_potential(Dispersion::new(-341345.136436851, -8));
    tail_far[3]
        .add_potential(Dispersion::new(-12287.2175217778, -7))
        .add_potential(Dispersion::new(-1015530.4019772, -9));
    tail_far[4]
        .add_potential(Dispersion::new(-51933.9885816, -8))
        .add_potential(Dispersion::new(-3746260.46991, -10));

    let mut exch_far = Vec::new();
    for _ in 0..=max_degree {
        exch_far.push((0., 0.));
    }

    let b_exch = Energy(f64::exp(15.847688), CmInv).to_au();
    let a_exch = -1.5090630 / Angstrom::TO_AU_MUL;
    exch_far[0] = (b_exch, a_exch);

    let b_exch = Energy(-f64::exp(16.3961123), CmInv).to_au();
    let a_exch = -1.508641657417 / Angstrom::TO_AU_MUL;
    exch_far[1] = (b_exch, a_exch);

    let b_exch = Energy(f64::exp(15.14425644), CmInv).to_au();
    let a_exch = -1.44547680 / Angstrom::TO_AU_MUL;
    exch_far[2] = (b_exch, a_exch);

    let b_exch = Energy(-f64::exp(12.53830479), CmInv).to_au();
    let a_exch = -1.33404298 / Angstrom::TO_AU_MUL;
    exch_far[3] = (b_exch, a_exch);

    let b_exch = Energy(f64::exp(9.100058), CmInv).to_au();
    let a_exch = -1.251990 / Angstrom::TO_AU_MUL;
    exch_far[4] = (b_exch, a_exch);

    let transition = |r, r_min, r_max| {
        if r <= r_min {
            1.
        } else if r >= r_max {
            0.
        } else {
            let x = ((r - r_max) - (r_min - r)) / (r_max - r_min);
            0.5 - 0.25 * f64::sin(PI / 2. * x) * (3. - f64::sin(PI / 2. * x).powi(2))
        }
    };

    let distances_extended = linspace(distances[0], 50., 500);
    let mut potentials_singlet = Vec::new();
    let mut potentials_triplet = Vec::new();
    let polynomials: Vec<Vec<f64>> = angles
        .iter()
        .map(|x| legendre_polynomials(max_degree, x.cos()))
        .collect();

    for lambda in 0..=max_degree {
        let tail = &tail_far[lambda as usize];
        let exch = exch_far[lambda as usize];

        let values_triplet = distances_extended
            .par_iter()
            .map(|x| {
                let value_rkhs: f64 = weights
                    .iter()
                    .zip(rkhs_triplet.iter())
                    .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                    .map(|((w, rkhs), p)| (lambda as f64 + 0.5) * w * p * rkhs.value(*x))
                    .sum();
                let value_far = tail.value(*x);

                let r_min = Distance(9., Angstrom).to_au();
                let r_max = Distance(11., Angstrom).to_au();

                let x = transition(*x, r_min, r_max);

                x * value_rkhs + (1. - x) * value_far
            })
            .collect::<Vec<f64>>();
        potentials_triplet.push((lambda as u32, values_triplet));

        let values_singlet = distances_extended
            .par_iter()
            .map(|x| {
                let value_rkhs: f64 = weights
                    .iter()
                    .zip(rkhs_triplet.iter())
                    .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                    .map(|((w, rkhs), p)| (lambda as f64 + 0.5) * w * p * rkhs.value(*x))
                    .sum();
                let value_far = tail.value(*x);

                let r_min = Distance(9., Angstrom).to_au();
                let r_max = Distance(11., Angstrom).to_au();

                let contrib = transition(*x, r_min, r_max);
                let triplet_part = contrib * value_rkhs + (1. - contrib) * value_far;

                let exch_rkhs: f64 = weights
                    .iter()
                    .zip(rkhs_exch.iter())
                    .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                    .map(|((w, rkhs), p)| (lambda as f64 + 0.5) * w * p * rkhs.value(*x))
                    .sum();
                let exch_far = exch.0 * f64::exp(exch.1 * x);

                let r_min = Distance(4.5, Angstrom).to_au();
                let r_max = if lambda == 0 {
                    Distance(6.5, Angstrom).to_au()
                } else if lambda < 4 {
                    Distance(7.5, Angstrom).to_au()
                } else {
                    Distance(5.5, Angstrom).to_au()
                };

                let contrib = transition(*x, r_min, r_max);
                let exch_contrib = contrib * exch_rkhs + (1. - contrib) * exch_far;

                triplet_part - exch_contrib
            })
            .collect::<Vec<f64>>();
        potentials_singlet.push((lambda as u32, values_singlet));
    }

    let singlets = PotentialArray {
        potentials: potentials_singlet,
        distances: distances_extended.clone(),
    };

    let triplets = PotentialArray {
        potentials: potentials_triplet,
        distances: distances_extended,
    };

    [singlets, triplets]
}

fn get_interpolated(
    pot_array: &PotentialArray,
) -> Vec<(u32, impl SimplePotential + Clone + use<>)> {
    let interp_potentials = interpolate_potentials(pot_array, 3);

    let mut potentials_far = Vec::new();
    for _ in &interp_potentials {
        potentials_far.push(Composite::new(Dispersion::new(0., 0)));
    }
    potentials_far[0]
        .add_potential(Dispersion::new(-3495.30040855597, -6))
        .add_potential(Dispersion::new(-516911.950541056, -8));
    potentials_far[1]
        .add_potential(Dispersion::new(17274.8363457991, -7))
        .add_potential(Dispersion::new(4768422.32042577, -9));
    potentials_far[2]
        .add_potential(Dispersion::new(-288.339392609436, -6))
        .add_potential(Dispersion::new(-341345.136436851, -8));
    potentials_far[3]
        .add_potential(Dispersion::new(-12287.2175217778, -7))
        .add_potential(Dispersion::new(-1015530.4019772, -9));
    potentials_far[4]
        .add_potential(Dispersion::new(-51933.9885816, -8))
        .add_potential(Dispersion::new(-3746260.46991, -10));

    let transition = |r| {
        if r <= 40. {
            1.
        } else if r >= 50. {
            0.
        } else {
            0.5 * (1. + f64::cos(PI * (r - 40.) / 10.))
        }
    };

    interp_potentials
        .into_iter()
        .zip(potentials_far.into_iter())
        .map(|((lambda, near), far)| {
            let combined = TransitionedPotential::new(near, far, transition);

            (lambda, combined)
        })
        .collect()
}
