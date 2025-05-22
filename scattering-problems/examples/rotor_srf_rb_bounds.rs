use std::time::Instant;

use clebsch_gordan::hi32;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::{
        MHz,
        energy_units::{Energy, GHz, Kelvin},
    },
    utility::linspace,
};
use scattering_problems::{
    FieldScatteringProblem,
    alkali_rotor_atom::{AlkaliRotorAtomProblemBuilder, TramBasisRecipe},
    rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder},
};
use scattering_solver::{
    log_derivatives::johnson::Johnson,
    numerovs::LocalWavelengthStepRule,
    observables::bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence},
    utility::save_serialize,
};

use rayon::prelude::*;
mod common;

use common::{PotentialType, ScalingType, Scalings, srf_rb_functionality::*};

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Bounds",
    "magnetic field bounds" => |_| Self::magnetic_field_bounds(),
    "potential surface scaling" => |_| Self::potential_surface_scaling(),
    "potential surface 2d scaling" => |_| Self::potential_surface_2d_scaling(),
    "magnetic field bounds" => |_| Self::magnetic_field_bounds_scaling(),
);

impl Problems {
    fn magnetic_field_bounds() {
        let projection = hi32!(1);
        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 500);
        let basis_recipe = TramBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };

        let energy_range = (Energy(-1., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        let scaling_singlet: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Isotropic, ScalingType::Anisotropic],
            scalings: vec![1.00354, 0.91387755],
        });
        let scaling_triplet: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Isotropic, ScalingType::Anisotropic],
            scalings: vec![1.0071, 0.8142857],
        });
        let suffix = "scaled_v0";

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
        let bound_states = mag_fields
            .iter()
            .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);

                let asymptotic = alkali_problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(energy_range, err)
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
            "SrF_Rb_bound_states_n_max_{}_{}",
            basis_recipe.n_max, suffix
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn potential_surface_scaling() {
        let potential_type = PotentialType::Singlet;
        let scaling_type = ScalingType::Full;

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 0,
            n_max: 0,
            ..Default::default()
        };
        let scalings = linspace(0.95, 1.05, 200);

        ///////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, hi32!(0));

        let [singlet, triplet] = read_extended(25);
        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };
        let pes = get_interpolated(&pes);

        let start = Instant::now();
        let singlet_bounds = scalings
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &scaling| {
                let pes = scaling_type.scale(&pes, scaling);

                let problem = RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: singlet_bounds,
        };
        let filename = format!(
            "SrF_Rb_bounds_{potential_type}_scaling_{scaling_type}_n_max_{}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn potential_surface_2d_scaling() {
        let potential_type = PotentialType::Singlet;
        let scaling_types = vec![
            ScalingType::Isotropic, 
            ScalingType::Anisotropic
        ];

        let energy_range = (Energy(-12., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings1 = linspace(1., 1.01, 50);
        let scalings2 = linspace(0.90, 0.92, 50);
        let suffix = "zoomed_0,9111";

        ///////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, hi32!(0));

        let [singlet, triplet] = read_extended(25);
        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };
        let pes = get_interpolated(&pes);

        let scalings: Vec<(f64, f64)> = scalings1
            .iter()
            .flat_map(|s1| scalings2.iter().map(|s2| (*s1, *s2)))
            .collect();

        let start = Instant::now();
        let pes_bounds = scalings
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, &(s1, s2)| {
                let scalings = Scalings {
                    scalings: vec![s1, s2],
                    scaling_types: scaling_types.clone(),
                };
                let pes = scalings.scale(&pes);

                let problem = RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: pes_bounds,
        };
        let filename = format!(
            "SrF_Rb_bounds_{potential_type}_2d_scaling_{}_{}_n_max_{}_{suffix}",
            scaling_types[0], scaling_types[1], basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn magnetic_field_bounds_scaling() {
        let mag_fields = linspace(0., 1000., 500);
        
        let potential_type = PotentialType::Singlet;
        let scaling_type = ScalingType::Full;
        let scalings = linspace(1., 1.02, 50);

        let other_scaling: Option<Scalings> = Some(Scalings {
            scaling_types: vec![ScalingType::Full],
            scalings: vec![1.0151],
        });

        let energy_range = (Energy(-1., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

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
        let bound_states = scaling_field
            .par_iter()
            .progress()
            .map_with(atoms, |atoms, (scaling, mag_field)| {
                let mut atoms = atoms.clone();

                let singlet = match potential_type {
                    PotentialType::Singlet => scaling_type.scale(&singlet, *scaling),
                    PotentialType::Triplet => if let Some(scaling) = &other_scaling {
                        scaling.scale(&singlet)
                    } else {
                        ScalingType::Full.scale(&singlet, 1.)
                    }
                };

                let triplet = match potential_type {
                    PotentialType::Triplet => scaling_type.scale(&triplet, *scaling),
                    PotentialType::Singlet => if let Some(scaling) = &other_scaling {
                        scaling.scale(&triplet)
                    } else {
                        ScalingType::Full.scale(&triplet, 1.)
                    },
                };

                let alkali_problem = AlkaliRotorAtomProblemBuilder::new(triplet, singlet)
                    .build(&atoms, &basis_recipe);

                let alkali_problem = alkali_problem.scattering_for(*mag_field);

                let asymptotic = alkali_problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect::<Vec<BoundStates>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scaling_field,
            bound_states,
        };

        let filename = format!(
            "SrF_Rb_bound_states_n_max_{}_{potential_type}_scaling_{scaling_type}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }
}
