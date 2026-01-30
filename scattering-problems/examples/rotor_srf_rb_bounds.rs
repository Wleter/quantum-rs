use std::{f64::consts::PI, sync::{Arc, Mutex}, time::Instant};

use argmin::{core::{CostFunction, Executor, State, observers::ObserverMode}, solver::{neldermead::NelderMead, simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing}}};
use argmin_observer_slog::SlogLogger;
use clebsch_gordan::hi32;
use hhmmss::Hhmmss;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::{
        Au, CmInv, MHz, energy_units::{Energy, GHz, Kelvin}
    },
    utility::linspace,
};
use rand::{Rng, SeedableRng, distr::Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use scattering_problems::{
    alkali_rotor_atom::{AlkaliRotorAtomProblemBuilder, TramBasisRecipe}, field_bound_states::{FieldBoundStates, FieldBoundStatesDependence, FieldProblemBuilder}, rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder}, FieldScatteringProblem
};
use scattering_solver::{
    log_derivatives::johnson::Johnson,
    numerovs::LocalWavelengthStepRule,
    observables::bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence, WaveFunctions},
    utility::{save_data, save_serialize},
};

use rayon::prelude::*;
mod common;

use common::{PotentialType, ScalingType, Scalings, srf_rb_functionality::*};

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
    "bound states reconstruction stochastic" => |_| Self::bound_states_reconstruction_stochastic(),
    "bound states reconstruction local" => |_| Self::bound_states_reconstruction_local(),
    "bound states reconstruction random" => |_| Self::bound_states_reconstruction_random(),
    "states density" => |_| Self::states_density(),
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
            scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
            scalings: vec![1.11114264265224, 0.989117739325556],
        };
        let scaling_singlet = Scalings {
            scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
            scalings: vec![1.0262636558798472, 0.8663308327900304],
        };
        let suffix = "scaled_1_11_0_99_1_02_0_87";

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
        let potential_type = PotentialType::Singlet;

        let energy_range = (Energy(-13., GHz), Energy(0., GHz));
        let err = Energy(1e-2, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scaling_c = 0.901;
        let scaling_d = 0.1;
        let scaling_no = 1001;
        
        let scalings = linspace(scaling_c, scaling_c+scaling_d, scaling_no);
        let calc_wave = true;
        let suffix = "invariant_scaling";
        let lambda_1 = 1.;
        let ratio = 4.5;

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

                let morph = Scalings {
                    scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
                    // todo! temporarily change scaling so that it counters n = 0 states shift
                    scalings: vec![scaling, lambda_1 - (scaling - scaling_c) * ratio] 
                };

                let pes = morph.scale(&pes);

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
            "SrF_Rb_bounds_{potential_type}_scaling_n_max_{}_{suffix}",
            basis_recipe.n_max
        );

        save_serialize(&filename, &data).unwrap();
    }
    
    fn potential_surface_scaling_field() {
        let potential_type = PotentialType::Singlet;
        let scaling_type = ScalingType::Full;

        for n_max in [33, 34, 35, 36, 37, 38] {
            let scaling_range = (1., 1.05);
            let err = 1e-6;

            let basis_recipe = RotorAtomBasisRecipe {
                l_max: n_max,
                n_max: n_max,
                ..Default::default()
            };
            let energies: Vec<Energy<GHz>> = linspace(-2.3, 0., 201)
                .iter()
                .map(|x| Energy(x.powi(3), GHz))
                .collect();
            let calc_wave = true;
            let suffix = "scaling";
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
                        let scalings = Scalings {
                            scalings: vec![scaling, lambda_aniso],
                            scaling_types: vec![ScalingType::Full, ScalingType::Anisotropic],
                        };
                        let pes = scalings.scale(&pes);
        
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
    }

    fn potential_surface_2d_scaling() {
        let potential_type = PotentialType::Triplet;

        let energy_range = (Energy(-13., GHz), Energy(0., GHz));
        let err = Energy(0.1, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings1 = linspace(0.96, 1.04, 201);
        let scalings2 = linspace(0.80, 1., 101);
        let suffix = "sr_scaling_1_00";
        let calc_wave = true;
        let potentials = RawRKHSLegendre::new(25);

        ///////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative, hi32!(0));

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
                let pes = get_interpolated(&potentials.get_scaled(potential_type, Some(&scalings)));

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
            scalings: vec![1.0601650899579287, 0.8021238213825383],
        };
        
        let energy_range = (Energy(-13., GHz), Energy(0., GHz));
        let err = Energy(0.1, MHz);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let suffix = "_scaling_test";

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

    fn bound_states_reconstruction_stochastic() {
        let potential_type = PotentialType::Singlet;

        let scaling_types = vec![ScalingType::Full, ScalingType::Anisotropic];
        let (center_0, center_1) = (1.0262636558798472, 0.8663308327900304);
        let (d_0, d_1) = (0.1, 0.2);
        let max_iter = 300;
        let temp_max = 15.0;

        let res = (0..12)
            .into_par_iter()
            .map(|i| {
                let reconstructing_bound = match potential_type {
                    PotentialType::Triplet => vec![
                        (0, Energy(-0.04527481, GHz), [0.991638424564369, 0.0021309375745617734, 0.0005737144038586135]),
                        (1, Energy(-0.77267861, GHz), [0.9576243160041955, 0.01263385544804692, 0.0030237962482492827]),
                        (2, Energy(-3.09381825, GHz), [0.8636080850631245, 0.047039854616814814, 0.007340606157280014]),
                        (3, Energy(-4.64858425, GHz), [0.1694511839839671, 0.4358310518008622, 0.06061805142594878]),
                        (4, Energy(-7.685404814086, GHz), [0.04898737032785113, 0.28474037829590915, 0.02364966636062396]),
                        (5, Energy(-8.225306960348, GHz), [0.7305319020669387, 0.08230229641246066, 0.022186879150599075]),
                        (6, Energy(-9.907725416783, GHz), [0.11200047532253958, 0.04480037197787622, 0.04226070770022582]),
                    ],
                    PotentialType::Singlet => vec![
                        (0, Energy(-0.1019540032144, GHz), [0.9836993, 0.00440876, 0.00204719]),
                        (1, Energy(-1.059685046795, GHz), [0.93178297, 0.01416468, 0.00833176]),
                        (2, Energy(-3.663536724117, GHz), [0.82649558, 0.02376356, 0.011734]),
                        (3, Energy(-3.78752205, GHz), [0.05866517, 0.04675238, 0.25008782]),
                        (4, Energy(-6.65379255529, GHz), [0.05548231, 0.44377141, 0.05224841]),
                        (5, Energy(-8.19758056726, GHz), [0.66102461, 0.05570395, 0.09580309]),
                    ]
                };

                let energy_range = (Energy(-13., GHz), Energy(0., GHz));
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
                    bounds: (vec![1. - d_0, 1. - d_1], vec![1. + d_0, 1.]),
                    rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_os_rng())),
                    temp_max
                };
        
                let solver = SimulatedAnnealing::new(temp_max).unwrap()
                    .with_temp_func(SATempFunc::Boltzmann)
                    .with_reannealing_best(30)
                    .with_reannealing_accepted(30)
                    .with_reannealing_fixed(100);
        
                let res = Executor::new(bound_reconstruction, solver)
                    .configure(|state| 
                        state.param(vec![center_0, center_1])
                            .max_iters(max_iter)
                            .target_cost(0.)
                )
                .add_observer(
                    SlogLogger::term_noblock(), 
                    ObserverMode::NewBest
                )
                .run()
                .unwrap();
        
                let best = res.state().get_best_param().unwrap();
                let chi2 = res.state().get_best_cost();
        
                println!("{i} {res}");
                println!("best scalings: {best:?}");
                println!("scaling types: {scaling_types:?}");
                println!("chi2: {chi2}");

                (best.clone(), chi2)
            }
        )
        .collect::<Vec<(Vec<f64>, f64)>>();

        let best_res = res.iter().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();

        println!("best overall");
        println!("best scalings: {:?}", best_res.0);
        println!("scaling types: {scaling_types:?}");
        println!("chi2: {}", best_res.1);
    }

    fn bound_states_reconstruction_local() {
        let potential_type = PotentialType::Singlet;

        let scaling_types = vec![ScalingType::Full, ScalingType::Anisotropic];
        let (center_0, center_1) = (1.0601650899579287, 0.8021238213825383);
        let (d_0, d_1) = (0.02, 0.01);
        let max_iter = 30;

        let reconstructing_bound = match potential_type {
            PotentialType::Triplet => vec![
                (0, Energy(-0.04527481, GHz), [0.991638424564369, 0.0021309375745617734, 0.0005737144038586135]),
                (1, Energy(-0.77267861, GHz), [0.9576243160041955, 0.01263385544804692, 0.0030237962482492827]),
                (2, Energy(-3.09381825, GHz), [0.8636080850631245, 0.047039854616814814, 0.007340606157280014]),
                (3, Energy(-4.64858425, GHz), [0.1694511839839671, 0.4358310518008622, 0.06061805142594878]),
                (4, Energy(-7.685404814086, GHz), [0.04898737032785113, 0.28474037829590915, 0.02364966636062396]),
                (5, Energy(-8.225306960348, GHz), [0.7305319020669387, 0.08230229641246066, 0.022186879150599075]),
                (6, Energy(-9.907725416783, GHz), [0.11200047532253958, 0.04480037197787622, 0.04226070770022582]),
            ],
            PotentialType::Singlet => vec![
                (0, Energy(-0.1019540032144, GHz), [0.9836993, 0.00440876, 0.00204719]),
                (1, Energy(-1.059685046795, GHz), [0.93178297, 0.01416468, 0.00833176]),
                (2, Energy(-3.663536724117, GHz), [0.82649558, 0.02376356, 0.011734]),
                (3, Energy(-3.78752205, GHz), [0.05866517, 0.04675238, 0.25008782]),
                (4, Energy(-6.65379255529, GHz), [0.05548231, 0.44377141, 0.05224841]),
                (5, Energy(-8.19758056726, GHz), [0.66102461, 0.05570395, 0.09580309]),
            ]
        };
        
        let energy_range = (Energy(-13., GHz), Energy(0., GHz));
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

        // let solver = ParticleSwarm::new(
        //     (
        //         vec![1. - d_0, 1. - d_1],
        //         vec![1. + d_0, 1.]
        //     ), 
        //     32
        // );

        let init_simplex = vec![
            vec![center_0, center_1], 
            vec![center_0 + d_0, center_1], 
            vec![center_0 - d_0, center_1 + d_1], 
        ];
        let solver = NelderMead::new(init_simplex);

        let bound_reconstruction = BoundMinimizationProblem {
            reconstructing_bound,
            scaling_types: scaling_types.clone(),
            calculation,
            bounds: (vec![1. - d_0, 1. - d_1], vec![1. + d_0, 1.]),
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_os_rng())),
            temp_max: 0.
        };

        let res = Executor::new(bound_reconstruction, solver)
            .configure(|state| 
                state.param(vec![center_0, center_1])
                    .max_iters(max_iter)
                    .target_cost(0.)
        )
        .add_observer(
            SlogLogger::term_noblock(), 
            ObserverMode::Always
        )
        .run()
        .unwrap();

        let best = res.state().get_best_param().unwrap();
        let chi2 = res.state().get_best_cost();

        println!("{res}");
        println!("best scalings: {best:?}");
        println!("scaling types: {scaling_types:?}");
        println!("chi2: {chi2}");
    }

    fn bound_states_reconstruction_random() {
        let potential_type = PotentialType::Singlet;

        let scaling_types = vec![ScalingType::Full, ScalingType::Anisotropic];
        let (d_0, d_1) = (0.1, 0.2);
        let max_iter = 30_000;

        let best = Mutex::new(None);
        let rng = Mutex::new(Xoshiro256PlusPlus::from_os_rng());

        (0..max_iter)
            .into_par_iter()
            .for_each(|_| {
                let scaling = {
                    let mut rng = rng.lock().unwrap();
                    let x = Uniform::new_inclusive(1. - d_0, 1. + d_0).unwrap();
                    let y = Uniform::new_inclusive(1. - d_1, 1.).unwrap();

                    let x = rng.sample(x);
                    let y = rng.sample(y);
                    vec![x, y]
                };

                let reconstructing_bound = match potential_type {
                    PotentialType::Triplet => vec![
                        (0, Energy(-0.04527481, GHz), [0.991638424564369, 0.0021309375745617734, 0.0005737144038586135]),
                        (1, Energy(-0.77267861, GHz), [0.9576243160041955, 0.01263385544804692, 0.0030237962482492827]),
                        (2, Energy(-3.09381825, GHz), [0.8636080850631245, 0.047039854616814814, 0.007340606157280014]),
                        (3, Energy(-4.64858425, GHz), [0.1694511839839671, 0.4358310518008622, 0.06061805142594878]),
                        (4, Energy(-7.685404814086, GHz), [0.04898737032785113, 0.28474037829590915, 0.02364966636062396]),
                        (5, Energy(-8.225306960348, GHz), [0.7305319020669387, 0.08230229641246066, 0.022186879150599075]),
                        (6, Energy(-9.907725416783, GHz), [0.11200047532253958, 0.04480037197787622, 0.04226070770022582]),
                    ],
                    PotentialType::Singlet => vec![
                        (0, Energy(-0.1019540032144, GHz), [0.9836993, 0.00440876, 0.00204719]),
                        (1, Energy(-1.059685046795, GHz), [0.93178297, 0.01416468, 0.00833176]),
                        (2, Energy(-3.663536724117, GHz), [0.82649558, 0.02376356, 0.011734]),
                        (3, Energy(-3.78752205, GHz), [0.05866517, 0.04675238, 0.25008782]),
                        (4, Energy(-6.65379255529, GHz), [0.05548231, 0.44377141, 0.05224841]),
                        (5, Energy(-8.19758056726, GHz), [0.66102461, 0.05570395, 0.09580309]),
                    ]
                };

                let energy_range = (Energy(-13., GHz), Energy(0., GHz));
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
                    bounds: (vec![1. - d_0, 1. - d_1], vec![1. + d_0, 1.]),
                    rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_os_rng())),
                    temp_max: 0.
                };

                let chi2 = bound_reconstruction.cost(&scaling).unwrap();
        
                let best: &mut Option<(Vec<f64>, f64)> = &mut best.lock().unwrap();
                match best {
                    Some(b) => if b.1 > chi2 {
                        b.0 = scaling;
                        b.1 = chi2;

                        println!("best scalings: {best:?}");
                        println!("scaling types: {scaling_types:?}");
                        println!("chi2: {chi2}");
                    } else if chi2 < 0.1 {
                        println!("best scalings: {scaling:?}");
                        println!("scaling types: {scaling_types:?}");
                        println!("chi2: {chi2}");
                    },
                    None => *best = Some((scaling, chi2)),
                }
            }
        );

        println!("{:?}", best.lock().unwrap().as_ref());
    }

    fn states_density() {
        let potential_type = PotentialType::Triplet;

        let energies = linspace(-3000., -0.5, 96);

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        };
        let scalings = Scalings {
            scaling_types: vec![ScalingType::Full],
            scalings: vec![1.0],
        };
        let suffix = "";

        ///////////////////////////////////

        let energy_relative = Energy(1e-7, Kelvin);
        let mut atoms = get_particles(energy_relative, hi32!(0));

        let [singlet, triplet] = read_extended(25);
        let pes = match potential_type {
            PotentialType::Singlet => singlet,
            PotentialType::Triplet => triplet,
        };
        let pes = get_interpolated(&pes);
        let pes = scalings.scale(&pes);

        let problem = RotorAtomProblemBuilder::new(pes.clone()).build(&atoms, &basis_recipe);
        let asymptotic = problem.asymptotic;
        atoms.insert(asymptotic);
        let potential = problem.potential;

        let bound_problem = BoundProblemBuilder::new(&atoms, &potential)
            .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
            .with_range(5., 20., 50.)
            .build();

        let start = Instant::now();
        let data: Vec<Vec<f64>> = energies
            .par_iter()
            .progress()
            .map(|&energy| {
                let energy = Energy(energy, CmInv);
                let mismatch = bound_problem.bound_mismatch(energy);

                vec![energy.value(), mismatch.nodes as f64]
            })
            .collect();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let filename = format!(
            "srf_rb_nodes_{potential_type}_n_max_{}{suffix}",
            basis_recipe.n_max
        );
        save_data(&filename, "energy\tnodes", &data).unwrap()
    }
}

struct BoundMinimizationProblem<F: Fn(Scalings) -> BoundStates> {
    reconstructing_bound: Vec<(u32, Energy<GHz>, [f64; 3])>,
    scaling_types: Vec<ScalingType>,
    calculation: F,
    bounds: (Vec<f64>, Vec<f64>),
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    temp_max: f64,
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
                if *index as usize >= bound_states.energies.len() {
                    1.
                } else {
                    let bound_state = bound_states.energies.get(*index as usize).unwrap();
    
                    let b_occupation = &bound_states.occupations.as_ref().unwrap()[*index as usize];
                    let occupation_chi2 = (occupation[0] - b_occupation[0]).powi(2)
                        + (occupation[1] - b_occupation[1]).powi(2)
                        + (occupation[2] - b_occupation[2]).powi(2);
    
                    (x.value() / bound_state).log2().powi(2) + occupation_chi2
                }

            })
            .sum();

        Ok(chi2 / (4. * self.reconstructing_bound.len() as f64) * 100.)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

impl<F: Fn(Scalings) -> BoundStates> Anneal for BoundMinimizationProblem<F> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Float = f64;

    fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<Self::Output, argmin_math::Error> {
        let mut param_n = param.clone();
        let mut rng = self.rng.lock().unwrap();
        let direction = Uniform::new_inclusive(0., 2. * PI)?;
        let val_distr = Uniform::new_inclusive(-0.2, 0.2)?;

        let dir = rng.sample(direction);
        let val = rng.sample(val_distr) * extent / self.temp_max;

        let width_x = self.bounds.1[0] - self.bounds.0[0];
        let width_y = self.bounds.1[1] - self.bounds.0[1];

        param_n[0] += dir.cos() * width_x * val;
        param_n[1] += dir.sin() * width_y * val;

        param_n[0] = param_n[0].clamp(self.bounds.0[0], self.bounds.1[0]);
        param_n[1] = param_n[1].clamp(self.bounds.0[1], self.bounds.1[1]);

        Ok(param_n)
    }
}