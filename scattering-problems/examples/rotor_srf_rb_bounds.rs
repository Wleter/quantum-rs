use std::time::Instant;

use clebsch_gordan::hi32;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    problem_selector::{get_args, ProblemSelector},
    problems_impl,
    units::{
        energy_units::{Energy, GHz, Kelvin}, MHz
    },
    utility::linspace,
};
use scattering_problems::{
    FieldScatteringProblem,
    alkali_rotor_atom::TramBasisRecipe,
    rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder},
};
use scattering_solver::{
    log_derivatives::johnson::Johnson, numerovs::LocalWavelengthStepRule, observables::bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence}, potentials::
        potential::ScaledPotential, utility::save_serialize
};

use rayon::prelude::*;
mod common;

use common::srf_rb_functionality::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "bound states" => |_| Self::bound_states(),
    "bound states potential scaling singlet" => |_| Self::bounds_singlet_scaling(),
    "bound states potential scaling triplet" => |_| Self::bounds_triplet_scaling(),
);

impl Problems {
    fn bound_states() {
        let entrance = 0;

        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 500);
        let basis_recipe = TramBasisRecipe {
            l_max: 0,
            n_max: 0,
            n_tot_max: 0,
            ..Default::default()
        };

        let atoms = get_particles(energy_relative, projection);
        let alkali_problem = get_problem(&atoms, &basis_recipe);

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let bound_states = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
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
        let scalings = linspace(0.95, 1.05, 200);

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let singlet_bounds = scalings
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &scaling| {
                let singlets = singlets
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

                let problem = RotorAtomProblemBuilder::new(singlets)
                    .build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = problem.potential;
    
                let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
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
        let filename = format!("SrF_Rb_singlet_bound_states_n_max_{}", basis_recipe.n_max);

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
        let scalings = linspace(0.95, 1.05, 200);

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
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
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