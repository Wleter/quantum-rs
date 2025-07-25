use std::time::Instant;

use clebsch_gordan::hi32;
use faer::Mat;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::energy_units::{Energy, Kelvin},
    utility::linspace,
};
use scattering_problems::{
    FieldScatteringProblem,
    alkali_rotor_atom::{AlkaliRotorAtomProblemBuilder, TramBasisRecipe},
    rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder},
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    log_derivatives::johnson::JohnsonLogDerivative,
    numerovs::{
        LocalWavelengthStepRule, multi_numerov::MultiRNumerov,
        propagator_watcher::PropagatorLogging,
    },
    observables::s_matrix::{ScatteringDependence, ScatteringObservables},
    potentials::potential::{Potential, ScaledPotential, SimplePotential},
    propagator::{CoupledEquation, Propagator},
    utility::{save_data, save_serialize, save_spectrum},
};

use rayon::prelude::*;
mod common;

use common::{PotentialType, ScalingType, Scalings, srf_rb_functionality::*};

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "potentials" => |_| Self::potentials(),
    "potential levels convergence" => |_| Self::potential_levels(),
    "single cross section calculation" => |_| Self::single_cross_sections(),
    "cross sections calculation" => |_| Self::cross_sections(),
    "a_length potential scaling" => |_| Self::potential_scaling_propagation(),
    "spinless convergence" => |_| Self::spinless_convergence(),
    "potential scaled scattering calculation" => |_| Self::scattering_scaled(),
    "scaled dependence scattering calculation" => |_| Self::magnetic_field_scattering_scaling(),
);

impl Problems {
    fn potentials() {
        let n_maxes = [0, 1, 5, 10, 20, 50];
        let distances = linspace(3., 80., 800);

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

        for n_max in n_maxes {
            let basis_recipe = RotorAtomBasisRecipe {
                l_max: n_max,
                n_max: n_max,
                ..Default::default()
            };

            let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
            let problem = RotorAtomProblemBuilder::new(interpolated.clone()).build(&atoms, &basis_recipe);
    
            let mut data = vec![];
            let mut potential_value = Mat::zeros(problem.potential.size(), problem.potential.size());
            for &r in &distances {
                problem.potential.value_inplace(r, &mut potential_value);
    
                data.push(
                    potential_value
                        .self_adjoint_eigenvalues(faer::Side::Lower)
                        .unwrap(),
                );
            }
    
            save_spectrum(
                &format!("SrF_Rb_triplet_adiabat_n_{}", basis_recipe.n_max),
                "distance\tadiabat",
                &distances,
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
    
            let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
            let problem = RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);
    
            let mut data = vec![];
            let mut potential_value = Mat::zeros(problem.potential.size(), problem.potential.size());
            for &r in &distances {
                problem.potential.value_inplace(r, &mut potential_value);
    
                data.push(
                    potential_value
                        .self_adjoint_eigenvalues(faer::Side::Lower)
                        .unwrap(),
                );
            }
    
            save_spectrum(
                &format!("SrF_Rb_singlet_adiabat_n_{}", basis_recipe.n_max),
                "distance\tadiabat",
                &distances,
                &data,
            )
            .unwrap();
        }

    }

    fn potential_levels() {
        let n_take = 5;
        let n_maxes: Vec<u32> = (n_take..100).collect();

        let distance = 40.;
        let levels_minima_triplet: Vec<Vec<f64>> = n_maxes
            .par_iter()
            .map(|&n| {
                let basis_recipe = RotorAtomBasisRecipe {
                    l_max: n,
                    n_max: n,
                    ..Default::default()
                };

                let [_, pot_array_triplet] = read_extended(25);
                let interpolated = get_interpolated(&pot_array_triplet);

                let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
                let problem =
                    RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);

                let mut potential_value = Mat::zeros(problem.potential.size(), problem.potential.size());
                problem.potential.value_inplace(distance, &mut potential_value);

                let levels = potential_value
                    .self_adjoint_eigenvalues(faer::Side::Lower)
                    .unwrap();
                let data = levels.into_iter().take(n_take as usize + 1).collect();

                data
            })
            .collect();

        let levels_minima_singlet: Vec<Vec<f64>> = n_maxes
            .iter()
            .map(|&n| {
                let basis_recipe = RotorAtomBasisRecipe {
                    l_max: n,
                    n_max: n,
                    ..Default::default()
                };

                let [pot_array_singlet, _] = read_extended(25);
                let interpolated = get_interpolated(&pot_array_singlet);

                let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
                let problem =
                    RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);

                let mut potential_value = Mat::zeros(problem.potential.size(), problem.potential.size());
                problem.potential.value_inplace(distance, &mut potential_value);

                let levels = potential_value
                    .self_adjoint_eigenvalues(faer::Side::Lower)
                    .unwrap();
                let data = levels.into_iter().take(n_take as usize + 1).collect();

                data
            })
            .collect();

        let n_maxes: Vec<f64> = n_maxes.iter().map(|&x| x as f64).collect();

        save_spectrum(
            &format!("SrF_Rb_triplet_adiabat_levels_r_{distance}"),
            "n_maxes\tminima",
            &n_maxes,
            &levels_minima_triplet,
        )
        .unwrap();

        save_spectrum(
            &format!("SrF_Rb_singlet_adiabat_levels_r_{distance}"),
            "n_maxes\tminima",
            &n_maxes,
            &levels_minima_singlet,
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
        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 500);
        let basis_recipe = TramBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };

        let scaling_singlet: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Isotropic, ScalingType::Anisotropic],
            scalings: vec![1.0036204085226377, 0.9129498323277407],
        });
        let scaling_triplet: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Isotropic, ScalingType::Anisotropic],
            scalings: vec![1.0069487290287622, 0.8152177020075073],
        });
        let suffix = "scaled_v1";

        ///////////////////////////////////

        let atoms = get_particles(energy_relative, projection);

        let [singlet, triplet] = read_extended(25);
        let singlet = get_interpolated(&singlet);
        let triplet = get_interpolated(&triplet);

        let triplet = if let Some(scalings) = &scaling_triplet {
            scalings.scale(&triplet)
        } else {
            ScalingType::Full.scale(&triplet, 1.)
        };

        let singlet = if let Some(scalings) = &scaling_singlet {
            scalings.scale(&singlet)
        } else {
            ScalingType::Full.scale(&singlet, 1.)
        };

        let alkali_problem =
            AlkaliRotorAtomProblemBuilder::new(triplet, singlet).build(&atoms, &basis_recipe);

        let start = Instant::now();
        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &mag_field| {
                let alkali_problem = alkali_problem.scattering_for(mag_field);
                atoms.insert(alkali_problem.asymptotic);
                let potential = &alkali_problem.potential;

                let boundary =
                    Boundary::new_multi_vanishing(5.0, Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(4e-3, 10., 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

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

        let filename = format!("SrF_Rb_scattering_n_max_{}_{}", basis_recipe.n_max, suffix);

        save_serialize(&filename, &data).unwrap()
    }

    fn magnetic_field_scattering_scaling() {
        let mag_fields = linspace(0., 1000., 500);

        let potential_type = PotentialType::Singlet;
        let scaling_type = ScalingType::Full;
        let scalings = linspace(1., 1.02, 50);

        let other_scaling: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Full],
            scalings: vec![1.0151],
        });

        let basis_recipe = TramBasisRecipe {
            l_max: 0,
            n_max: 0,
            ..Default::default()
        };

        ///////////////////////////////////

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, projection);

        let [singlet, triplet] = read_extended(25);
        let singlet = get_interpolated(&singlet);
        let triplet = get_interpolated(&triplet);

        let scaling_field: Vec<(f64, f64)> = scalings
            .iter()
            .flat_map(|s1| mag_fields.iter().map(|s2| (*s1, *s2)))
            .collect();

        let start = Instant::now();
        let scatterings = scaling_field
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, (scaling, mag_field)| {
                let mut atoms = atoms.clone();

                let singlet = match potential_type {
                    PotentialType::Singlet => scaling_type.scale(&singlet, *scaling),
                    PotentialType::Triplet => {
                        if let Some(scaling) = &other_scaling {
                            scaling.scale(&singlet)
                        } else {
                            ScalingType::Full.scale(&singlet, 1.)
                        }
                    }
                };

                let triplet = match potential_type {
                    PotentialType::Triplet => scaling_type.scale(&triplet, *scaling),
                    PotentialType::Singlet => {
                        if let Some(scaling) = &other_scaling {
                            scaling.scale(&triplet)
                        } else {
                            ScalingType::Full.scale(&triplet, 1.)
                        }
                    }
                };

                let alkali_problem = AlkaliRotorAtomProblemBuilder::new(triplet, singlet)
                    .build(&atoms, &basis_recipe);

                let alkali_problem = alkali_problem.scattering_for(*mag_field);

                let asymptotic = alkali_problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let boundary =
                    Boundary::new_multi_vanishing(5.0, Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(4e-3, 10., 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = ScatteringDependence {
            parameters: scaling_field,
            observables: scatterings,
        };

        let filename = format!(
            "SrF_Rb_scatterings_n_max_{}_{potential_type}_scaling_{scaling_type}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }
}
