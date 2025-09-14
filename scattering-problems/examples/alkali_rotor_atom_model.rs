use std::time::Instant;

use clebsch_gordan::hi32;
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use quantum::{problem_selector::{get_args, ProblemSelector}, problems_impl, units::{Au, Energy, GHz, Kelvin}, utility::linspace};
use scattering_problems::{field_bound_states::{FieldBoundStates, FieldBoundStatesDependence, FieldProblemBuilder}, rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder}};
use scattering_solver::{boundary::{Boundary, Direction}, log_derivatives::johnson::Johnson, numerovs::{multi_numerov::MultiRNumerov, LocalWavelengthStepRule}, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::{Potential, SimplePotential}}, propagator::{CoupledEquation, Propagator}, utility::{save_data, save_serialize, save_spectrum}};

use crate::common::{srf_rb_functionality::get_particles, PotentialType, ScalingType, Scalings};
use rayon::prelude::*;

mod common;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "alkali like rotor + atom model",
    "potentials" => |_| Self::potentials(),
    "basis convergence" => |_| Self::basis_convergence(),
    "pes scaling bound states" => |_| Self::scaling_bound_states()
);

impl Problems {
    fn potentials() {
        let n_maxes = [0, 1, 5, 10, 20, 50];
        let distances = linspace(3., 80., 800);

        let triplet = get_triplet();
        let singlet = get_singlet();

        let mut data = vec![distances.clone()];
        for (_, p) in &triplet {
            let values = distances.iter().map(|&x| p.value(x)).collect();

            data.push(values);
        }

        save_data(
            "rotor_atom_model_triplet_dec",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();

        let mut data = vec![distances.clone()];
        for (_, p) in &singlet {
            let values = distances.iter().map(|&x| p.value(x)).collect();

            data.push(values);
        }

        save_data(
            "rotor_atom_model_singlet_dec",
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
            let problem = RotorAtomProblemBuilder::new(triplet.clone()).build(&atoms, &basis_recipe);
    
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
                &format!("rotor_atom_model_triplet_adiabat_n_{}", basis_recipe.n_max),
                "distance\tadiabat",
                &distances,
                &data,
            )
            .unwrap();
    
            let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
            let problem = RotorAtomProblemBuilder::new(singlet.clone()).build(&atoms, &basis_recipe);
    
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
                &format!("rotor_atom_model_singlet_adiabat_n_{}", basis_recipe.n_max),
                "distance\tadiabat",
                &distances,
                &data,
            )
            .unwrap();
        }
    }

    fn basis_convergence() {
        let n_maxes: Vec<u32> = (0..200).collect();
        let energy_relative = Energy(1e-7, Kelvin);

        let singlets = get_singlet();
        let triplets = get_triplet();

        let atoms = get_particles(energy_relative, hi32!(0));

        let [triplet_scattering, singlet_scattering] = [PotentialType::Triplet, PotentialType::Singlet]
            .map(|spin| {
                let pes = match spin {
                    PotentialType::Triplet => &triplets,
                    PotentialType::Singlet => &singlets,
                };

                let start = Instant::now();
                let scattering = n_maxes
                    .par_iter()
                    .progress()
                    .map_with(atoms.clone(), |atoms, &n_max| {
                        let basis_recipe = RotorAtomBasisRecipe {
                            l_max: n_max,
                            n_max: n_max,
                            ..Default::default()
                        };

                        let problem =
                            RotorAtomProblemBuilder::new(pes.clone()).build(&atoms, &basis_recipe);

                        let asymptotic = problem.asymptotic;
                        atoms.insert(asymptotic);
                        let potential = &problem.potential;

                        let id = Mat::<f64>::identity(potential.size(), potential.size());
                        let boundary = Boundary::new(6.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));
                        let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
                        let eq = CoupledEquation::from_particles(potential, &atoms);
                        let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);
                        numerov.propagate_to(1500.);

                        numerov.s_matrix().get_scattering_length().re
                    })
                    .collect();

                    let elapsed = start.elapsed();
                    println!("calculated in {}", elapsed.hhmmssxxx());

                scattering
            });

        let n_maxes = n_maxes.into_iter().map(|x| x as f64).collect();
        let data = vec![n_maxes, singlet_scattering, triplet_scattering];

        save_data(
            &format!("rotor_atom_model_pes_scattering_convergence"),
            "scaling\tsinglet\ttriplet",
            &data,
        )
        .unwrap();
    }    
    
    fn scaling_bound_states() {
        let potential_type = PotentialType::Triplet;

        let scaling_range = (0.8, 1.2);
        let err = 1e-5;

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let energies: Vec<Energy<GHz>> = linspace(-2., 0., 101)
            .iter()
            .map(|x| Energy(x.powi(3), GHz))
            .collect();
        let calc_wave = true;
        let suffix = "";

        ///////////////////////////////////

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, projection);

        let singlet = get_singlet();
        let triplet = get_triplet();

        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };

        let start = Instant::now();
        let bounds: Vec<FieldBoundStates> = energies
            .par_iter()
            .progress()
            .map(|&energy| {
                let mut atoms = atoms.clone();
                atoms.insert(energy.to(Au));

                let morphed_problem = |scaling| {
                    let scaling = Scalings {
                        scaling_types: vec![ScalingType::Full],
                        scalings: vec![scaling],
                    };
                    let pes = scaling.scale(&pes);
    
                    RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe)
                };

                let bound_problem = FieldProblemBuilder::new(&atoms, &morphed_problem)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(6., 20., 500.)
                    .build();

                let mut bounds = bound_problem
                    .bound_states(scaling_range, err);

                if calc_wave {
                    let waves: Vec<Vec<f64>> = bound_problem.bound_waves(&bounds)
                        .map(|x| x.occupations())
                        .collect();
    
                    bounds.occupations = Some(waves);
                }

                bounds
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = FieldBoundStatesDependence {
            energies: energies.iter().map(|x| x.value()).collect(),
            bound_states: bounds,
        };
        let filename = format!(
            "rotor_atom_model_{potential_type}_scaling_n_max_{}{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap();
    }
}

fn get_triplet() -> Vec<(u32, Composite<Dispersion>)> {
    let v_0 = Composite::from_vec(vec![
        Dispersion::new(6.5e7, -10),
        Dispersion::new(-3495.30040855597, -6),
        Dispersion::new(-516911.950541056, -8)
    ]);

    let v_1 = Composite::from_vec(vec![
        Dispersion::new(-17274.8363457991, -7),
        Dispersion::new(-3068422.32042577, -9)
    ]);


    vec![
        (0, v_0),
        (1, v_1)
    ]
}

fn get_singlet() -> Vec<(u32, Composite<Dispersion>)> {
    let v_0 = Composite::from_vec(vec![
        Dispersion::new(2e9, -12),
        Dispersion::new(-3495.30040855597, -6),
        Dispersion::new(-516911.950541056, -8)
    ]);

    let v_1 = Composite::from_vec(vec![
        Dispersion::new(-17274.8363457991, -7),
        Dispersion::new(-768422.32042577, -9)
    ]);


    vec![
        (0, v_0),
        (1, v_1)
    ]
}