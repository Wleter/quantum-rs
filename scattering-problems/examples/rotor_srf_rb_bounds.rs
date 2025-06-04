use std::time::Instant;

use argmin::{core::{observers::ObserverMode, CostFunction, Executor, State}, solver::neldermead::NelderMead};
use argmin_observer_slog::SlogLogger;
use clebsch_gordan::hi32;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    problem_selector::{get_args, ProblemSelector},
    problems_impl,
    units::{
        energy_units::{Energy, GHz, Kelvin}, Au, MHz
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
    "bound states reconstruction" => |_| Self::bound_states_reconstruction(),
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
        let scaling_types = vec![ScalingType::Isotropic, ScalingType::Anisotropic];

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

    fn bound_states_reconstruction() {
        // let potential_type = PotentialType::Triplet;
        // let reconstructing_bound = vec![
        //     (0, Energy(-0.04527481, GHz)),
        //     (1, Energy(-0.77267861, GHz)),
        //     (2, Energy(-3.09381825, GHz)),
        //     (3, Energy(-4.64858425, GHz)),
        //     // (4, Energy(-7.68292606, GHz)),
        //     // (5, Energy(-8.22490901, GHz)),
        //     // (6, Energy(-9.90615487, GHz)),
        // ];

        let potential_type = PotentialType::Singlet;
        let reconstructing_bound = vec![
            (0, Energy(-0.13956298, GHz)),
            (1, Energy(-1.15303618, GHz)),
            (2, Energy(-1.56510323, GHz)),
            (3, Energy(-3.78752205, GHz)),
            // (4, Energy(-6.31701475, GHz)),
            // (5, Energy(-7.95295992, GHz)),
            // (6, Energy(-10.25863621, GHz)),
            // (7, Energy(-11.96461586, GHz)),
        ];

        let scaling_types = vec![ScalingType::Isotropic, ScalingType::Anisotropic];

        let (center_iso, center_aniso) = (1.1, 0.8);
        let (d_iso, d_aniso) = (0.1, 0.1);

        let max_iter = 500;

        let energy_range = (Energy(-6., GHz), Energy(0., GHz));
        let err = Energy(1., MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };

        /////////////////////////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);

        let [singlet, triplet] = read_extended(25);
        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };
        let pes = get_interpolated(&pes);

        let calculation = |scalings: Scalings| {
            // println!("{scalings:?}");
            let mut atoms = get_particles(energy_relative, hi32!(0));
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
                .energies
                .iter()
                .map(|x| Energy(*x, Au).to(GHz))
                .collect()
        };

        let bound_reconstruction = BoundMinimizationProblem {
            reconstructing_bound,
            scaling_types: scaling_types.clone(),
            calculation,
        };

        let init_simplex = vec![
            vec![center_iso, center_aniso], 
            vec![center_iso + d_iso, center_aniso], 
            vec![center_iso, center_aniso + d_aniso], 
        ];
        let solver = NelderMead::new(init_simplex);

        // let solver = ParticleSwarm::new(
        //     (
        //         vec![center_iso - d_iso, center_aniso - d_aniso],
        //         vec![center_iso + d_iso, center_aniso + d_aniso]
        //     ), 
        //     32
        // );

        let res = Executor::new(bound_reconstruction, solver)
            .configure(|state| 
                state.max_iters(max_iter)
                    .target_cost(0.)
        )
        .add_observer(
            SlogLogger::term(), 
            ObserverMode::Always
        )
        .run()
        .unwrap();

        let best = res.state().get_best_param().unwrap();
        let chi2 = res.state().get_best_cost();

        println!("{}", res);
        println!("best scalings: {best:?}");
        println!("scaling types: {scaling_types:?}");
        println!("chi2: {chi2}");
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

struct BoundMinimizationProblem<F: Fn(Scalings) -> Vec<Energy<GHz>>> {
    reconstructing_bound: Vec<(u32, Energy<GHz>)>,
    scaling_types: Vec<ScalingType>,
    calculation: F
}

impl<F: Fn(Scalings) -> Vec<Energy<GHz>>> CostFunction for BoundMinimizationProblem<F> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let scalings = Scalings {
            scaling_types: self.scaling_types.clone(),
            scalings: param.clone()
        };

        let mut energies = (self.calculation)(scalings);
        energies.sort_by(|a, b| b.value().partial_cmp(&a.value()).unwrap());

        let chi2: f64 = self.reconstructing_bound.iter()
            .map(|(index, x)| {
                let bound_state = energies.get(*index as usize).map_or(2. * x.value(), |x| x.value());

                (x.value() / bound_state).log2().powi(2)
            })
            .sum();

        Ok(chi2)
    }

    fn parallelize(&self) -> bool {
        true
    }
}
