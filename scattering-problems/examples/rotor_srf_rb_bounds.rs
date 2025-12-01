use std::time::Instant;

use argmin::{core::{CostFunction, Executor, State, observers::ObserverMode}, solver::neldermead::NelderMead};
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
    alkali_rotor_atom::{AlkaliRotorAtomProblemBuilder, TramBasisRecipe}, field_bound_states::{FieldBoundStates, FieldBoundStatesDependence, FieldProblemBuilder}, rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder}, FieldScatteringProblem
};
use scattering_solver::{
    log_derivatives::johnson::Johnson,
    numerovs::LocalWavelengthStepRule,
    observables::bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence, WaveFunctions},
    utility::save_serialize,
};

use rayon::prelude::*;
mod common;

use common::{PotentialType, ScalingType, Scalings, srf_rb_functionality::*};

use crate::common::Morphing;

pub fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(12).build_global().unwrap();
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Bounds",
    "magnetic field bounds" => |_| Self::magnetic_field_bounds(),
    "potential surface scaling" => |_| Self::potential_surface_scaling(),
    "potential surface scaling field" => |_| Self::potential_surface_scaling_field(),
    "potential surface 2d scaling" => |_| Self::potential_surface_2d_scaling(),
    "singlet/triplet bound waves" => |_| Self::bound_waves(),
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
        let err = Energy(0.1, MHz);

        let scaling_triplet = Scalings {
            scaling_types: vec![ScalingType::Isotropic, ScalingType::Anisotropic],
            scalings: vec![1.0069487290287622, 0.8152177020075073],
        };
        let scaling_singlet = Scalings {
            scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
            scalings: vec![0.9389302523757846, 0.976730434834512],
        };
        let suffix = "scaled_0_94_0_98";

        ///////////////////////////////////

        let atoms = get_particles(energy_relative, projection);

        let [singlet, triplet] = read_extended(25);
        let singlet = get_interpolated(&singlet);
        let triplet = get_interpolated(&triplet);

        let triplet = scaling_triplet.scale(&triplet);
        let singlet = scaling_singlet.scale(&singlet);

        let alkali_problem = AlkaliRotorAtomProblemBuilder::new(triplet, singlet).build(&atoms, &basis_recipe);

        let start = Instant::now();
        let bound_states = mag_fields
            .par_iter()
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
        let potential_type = PotentialType::Triplet;
        let scaling_type = ScalingType::Full;

        let energy_range = (Energy(-9., GHz), Energy(0., GHz));
        let err = Energy(1e-2, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scaling_c = 1.003;
        let scaling_d = 0.2;
        let scaling_no = 2_001;
    
        let scalings = linspace(scaling_c-scaling_d, scaling_c+scaling_d, scaling_no);
        let calc_wave = true;
        let suffix = "invariant";
        let lambda_1 = 0.;

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
        let bounds: Vec<BoundStates> = scalings
            .par_iter()
            .progress()
            .map(|&scaling| {
                let mut atoms = atoms.clone();

                let morph = Morphing {
                    lambdas: vec![0, 1],
                    // todo! temporarily change scaling so that it counters n = 0 states shift
                    scalings: vec![scaling, lambda_1 + (scaling - scaling_c) * 26. / 20.] 
                };

                let pes = morph.morph(&pes);

                let problem = RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe);

                let asymptotic = problem.asymptotic;
                atoms.insert(asymptotic);
                let potential = problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
                    .build();

                let mut bounds = bound_problem
                    .bound_states(energy_range, err);

                if calc_wave {
                    let waves: Vec<Vec<f64>> = bound_problem.bound_waves(&bounds)
                        .map(|x| x.occupations())
                        .collect();
    
                    bounds.occupations = Some(waves);
                }

                bounds.with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: bounds,
        };
        let filename = format!(
            "SrF_Rb_bounds_{potential_type}_scaling_{scaling_type}_n_max_{}_{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap();
    }
    
    fn potential_surface_scaling_field() {
        let potential_type = PotentialType::Singlet;
        let scaling_type = ScalingType::Full;

        let scaling_range = (1. - 0.1, 1. + 0.1);
        let err = 1e-6;

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let energies: Vec<Energy<GHz>> = linspace(-2.3, 0., 201)
            .iter()
            .map(|x| Energy(x.powi(3), GHz))
            .collect();
        let calc_wave = true;
        let suffix = "scaling_1_00";
        let lambda_aniso = 1.;

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
        let bounds: Vec<FieldBoundStates> = energies
            .par_iter()
            .progress()
            .map(|&energy| {
                let mut atoms = atoms.clone();
                atoms.insert(energy.to(Au));

                let morphed_problem = |scaling| {
                    let morph = Scalings {
                        scalings: vec![scaling, lambda_aniso],
                        scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
                    };
                    let pes = morph.scale(&pes);
    
                    RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe)
                };

                let bound_problem = FieldProblemBuilder::new(&atoms, &morphed_problem)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(5., 20., 500.)
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
            "SrF_Rb_field_{potential_type}_scaling_{scaling_type}_n_max_{}_{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap();
    }

    fn potential_surface_2d_scaling() {
        let potential_type = PotentialType::Singlet;

        let energy_range = (Energy(-8., GHz), Energy(0., GHz));
        let err = Energy(0.1, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings1 = linspace(1.41, 1.45, 101);
        let scalings2 = linspace(0.85, 1., 81);
        let suffix = "scaling_1_43";
        let calc_wave = true;

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
                    scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
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

                let mut bounds = bound_problem
                    .bound_states(energy_range, err);

                if calc_wave {
                    let waves: Vec<Vec<f64>> = bound_problem.bound_waves(&bounds)
                        .map(|x| x.occupations())
                        .collect();
    
                    bounds.occupations = Some(waves);
                }

                bounds.with_energy_units(GHz)
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let data = BoundStatesDependence {
            parameters: scalings,
            bound_states: pes_bounds,
        };
        let filename = format!(
            "SrF_Rb_bounds_{potential_type}_2d_scaling_n_max_{}_{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn bound_waves() {
        let potential_type = PotentialType::Singlet;
        let scaling = Scalings {
            scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
            scalings: vec![1.0134369126941278, 1.0096475412543842],
        };
        
        let energy_range = (Energy(-10., GHz), Energy(0., GHz));
        let err = Energy(0.1, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let suffix = "_scaling_1_01_1_00";

        ///////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);
        let mut atoms = get_particles(energy_relative, hi32!(0));

        let [singlet, triplet] = read_extended(25);
        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };
        let pes = get_interpolated(&pes);
        let pes = scaling.scale(&pes);

        let problem = RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe);

        let asymptotic = problem.asymptotic;
        atoms.insert(asymptotic);
        let potential = problem.potential;

        let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
            .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
            .with_range(5., 20., 500.)
            .build();

        let bounds = bound_problem
            .bound_states(energy_range, err);

        let waves = bound_problem.bound_waves(&bounds).collect();

        let data = WaveFunctions {
            bounds,
            waves,
        };

        let filename = format!(
            "SrF_Rb_bounds_{potential_type}_wave_functions_n_max_{}{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap()
    }

    fn bound_states_reconstruction() {
        // let potential_type = PotentialType::Triplet;
        // let reconstructing_bound = vec![
        //     (0, Energy(-0.04527481, GHz), [0.991638424564369, 0.0021309375745617734, 0.0005737144038586135]),
        //     (1, Energy(-0.77267861, GHz), [0.9576243160041955, 0.01263385544804692, 0.0030237962482492827]),
        //     (2, Energy(-3.09381825, GHz), [0.8636080850631245, 0.047039854616814814, 0.007340606157280014]),
        //     (3, Energy(-4.64858425, GHz), [0.1694511839839671, 0.4358310518008622, 0.06061805142594878]),
        //     (4, Energy(-7.685404814086, GHz), [0.04898737032785113, 0.28474037829590915, 0.02364966636062396]),
        //     (5, Energy(-8.225306960348, GHz), [0.7305319020669387, 0.08230229641246066, 0.022186879150599075]),
        //     (6, Energy(-9.907725416783, GHz), [0.11200047532253958, 0.04480037197787622, 0.04226070770022582]),
        // ];

        let potential_type = PotentialType::Singlet;
        let reconstructing_bound = vec![
            (0, Energy(-0.13956298, GHz), [0.9521102770332074, 0.005000084345797367, 0.008731931540568756]),
            (1, Energy(-1.15303618, GHz), [0.8829877255109269, 0.023721801610900935, 0.025887204644646177]),
            (2, Energy(-1.56510323, GHz), [0.0476238044176934, 0.3247536135778192, 0.18387629094793698]),
            (3, Energy(-3.78752205, GHz), [0.8189536928653838, 0.026488837868052114, 0.01707929422850362]),
            (4, Energy(-6.327935219094, GHz), [0.05843930881412625, 0.09759951597507624, 0.0850313321453373]),
            (5, Energy(-7.962288439908, GHz), [0.47397652910820515, 0.17791882445793714, 0.010912929190178784]),
            (6, Energy(-10.26483438346, GHz), [0.40379875237332147, 0.2854688458287785, 0.04217334676609967]),
            (7, Energy(-11.97014558676, GHz), [0.03459635689854624, 0.04614802295395822, 0.39359017460339235]),
            
        ];

        let scaling_types = vec![ScalingType::Full, ScalingType::Anisotropic];
        let (center_0, center_1) = (1.0134369126941278, 1.0096475412543842);
        let (d_0, d_1) = (0.01, 0.01);

        let max_iter = 30;

        let energy_range = (Energy(-6., GHz), Energy(0., GHz));
        let err = Energy(0.1, MHz);

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
            println!("{scalings:?}");
            let mut atoms = get_particles(energy_relative, hi32!(0));

            let morphing = Scalings {
                scaling_types: scaling_types.clone(),
                scalings: scalings.scalings,
            };

            let pes = morphing.scale(&pes);

            let problem = RotorAtomProblemBuilder::new(pes).build(&atoms, &basis_recipe);

            let asymptotic = problem.asymptotic;
            atoms.insert(asymptotic);
            let potential = problem.potential;

            let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
                .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                .with_range(5., 20., 500.)
                .build();

            let mut bound_states = bound_problem
                .bound_states(energy_range, err);

            let waves: Vec<Vec<f64>> = bound_problem.bound_waves(&bound_states)
                .map(|x| x.occupations())
                .collect();
    
            bound_states.occupations = Some(waves);

            bound_states
        };

        let bound_reconstruction = BoundMinimizationProblem {
            reconstructing_bound,
            scaling_types: scaling_types.clone(),
            calculation,
        };

        let init_simplex = vec![
            vec![center_0, center_1], 
            vec![center_0 + d_0, center_1], 
            vec![center_0, center_1 + d_1], 
        ];
        let solver = NelderMead::new(init_simplex);

        // let solver = ParticleSwarm::new(
        //     (
        //         vec![center_0 - d_0, center_1 - d_1],
        //         vec![center_0 + d_0, center_1 + d_1]
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
        let err = Energy(0.1, MHz);

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

struct BoundMinimizationProblem<F: Fn(Scalings) -> BoundStates> {
    reconstructing_bound: Vec<(u32, Energy<GHz>, [f64; 3])>,
    scaling_types: Vec<ScalingType>,
    calculation: F
}

impl<F: Fn(Scalings) -> BoundStates> CostFunction for BoundMinimizationProblem<F> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let scalings = Scalings {
            scaling_types: self.scaling_types.clone(),
            scalings: param.clone()
        };

        let bound_states = (self.calculation)(scalings).with_energy_units(GHz);

        let chi2: f64 = self.reconstructing_bound.iter()
            .map(|(index, x, occupation)| {
                let bound_state = bound_states.energies.get(*index as usize).map_or(2. * x.value(), |x| *x);

                let b_occupation = &bound_states.occupations.as_ref().unwrap()[*index as usize];
                let occupation_chi2 = (occupation[0] - b_occupation[0]).powi(2)
                    + (occupation[1] - b_occupation[1]).powi(2)
                    + (occupation[2] - b_occupation[2]).powi(2);

                (x.value() / bound_state).log2().powi(2) + occupation_chi2 / 10.
            })
            .sum();

        Ok(chi2 / self.reconstructing_bound.len() as f64)
    }

    fn parallelize(&self) -> bool {
        true
    }
}
