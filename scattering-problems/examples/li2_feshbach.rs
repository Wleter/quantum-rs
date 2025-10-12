use std::time::Instant;

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder, Symmetry};
use clebsch_gordan::{half_integer::HalfI32, hi32, hu32};
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use num::complex::Complex64;
use quantum::{
    params::{particle_factory, particles::Particles},
    problem_selector::{get_args, ProblemSelector},
    problems_impl,
    units::{
        energy_units::{Energy, Kelvin, MHz}, Au, GHz
    },
    utility::linspace,
};
use scattering_problems::{
    alkali_atoms::AlkaliAtomsProblemBuilder, field_bound_states::{FieldBoundStates, FieldBoundStatesDependence, FieldProblemBuilder}, IndexBasisDescription, ScatteringProblem
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    log_derivatives::johnson::{Johnson, JohnsonLogDerivative},
    numerovs::LocalWavelengthStepRule,
    observables::bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence},
    potentials::{
        composite_potential::Composite,
        dispersion_potential::Dispersion,
        potential::{MatPotential, Potential},
    },
    propagator::{CoupledEquation, Propagator},
    utility::{save_data, save_serialize},
};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Li2 Feshbach",
    "potential values" => |_| Self::potential_values(),
    "feshbach resonance" => |_| Self::feshbach(),
    "bound states" => |_| Self::bound_states(),
    "field bound states" => |_| Self::field_bound_states(),
);

impl Problems {
    fn get_problem(
        projection: HalfI32,
        mag_field: f64,
    ) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
        let first = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1))
            .with_hyperfine_coupling(Energy(228.2 / 1.5, MHz).to_au());

        let hifi_problem = DoubleHifiProblemBuilder::new_homo(first, Symmetry::Fermionic)
            .with_projection(projection);

        let mut li2_singlet = Composite::new(Dispersion::new(-1381., -6));
        li2_singlet.add_potential(Dispersion::new(1.112e7, -12));

        let mut li2_triplet = Composite::new(Dispersion::new(-1381., -6));
        li2_triplet.add_potential(Dispersion::new(2.19348e8, -12));

        AlkaliAtomsProblemBuilder::new(hifi_problem, li2_singlet, li2_triplet).build(mag_field)
    }

    fn get_particles() -> Particles {
        let li_first = particle_factory::create_atom("Li6").unwrap();
        let li_second = particle_factory::create_atom("Li6").unwrap();

        Particles::new_pair(li_first, li_second, Energy(1e-7, Kelvin))
    }

    fn potential_values() {
        let alkali_problem = Self::get_problem(hi32!(0), 100.);
        let potential = &alkali_problem.potential;

        let mut potential_mat = Mat::<f64>::identity(potential.size(), potential.size());

        let distances = linspace(4., 200., 1000);

        let mut p1 = Vec::new();
        let mut p2 = Vec::new();
        let mut p12 = Vec::new();
        let mut p21 = Vec::new();
        for &distance in distances.iter().progress() {
            potential.value_inplace(distance, &mut potential_mat);
            p1.push(potential_mat[(0, 0)]);
            p2.push(potential_mat[(1, 1)]);
            p12.push(potential_mat[(0, 1)]);
            p21.push(potential_mat[(1, 0)]);
        }

        let header = "mag_field\tptoentials";
        let data = vec![distances, p1, p2, p12, p21];

        save_data("li2_potentials", header, &data).unwrap()
    }

    fn feshbach() {
        ///////////////////////////////////

        let projection = hi32!(0);

        let mut mag_fields = linspace(0., 620., 620);
        mag_fields.append(&mut linspace(620., 625., 500));
        mag_fields.append(&mut linspace(625., 1200., 575));

        ///////////////////////////////////

        let start = Instant::now();

        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let alkali_problem = Self::get_problem(projection, mag_field);

                let mut li2 = Self::get_particles();
                let potential = &alkali_problem.potential;
                li2.insert(alkali_problem.asymptotic);

                let boundary =
                    Boundary::new_multi_vanishing(4., Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.);
                let eq = CoupledEquation::from_particles(potential, &li2);
                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

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

        save_data("li2_scatterings", header, &data).unwrap()
    }

    fn bound_states() {
        ///////////////////////////////////

        let projection = hi32!(0);
        let mag_fields = linspace(0., 1200., 1200);
        let energy_range = (Energy(-12., GHz), Energy(0., MHz));
        let err = Energy(1., MHz);

        ///////////////////////////////////

        let start = Instant::now();
        let bound_states = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let alkali_problem = Self::get_problem(projection, mag_field);

                let mut particles = Self::get_particles();
                let potential = &alkali_problem.potential;
                particles.insert(alkali_problem.asymptotic);

                let bound_problem = BoundProblemBuilder::new(&particles, potential)
                    .with_propagation(
                        LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.),
                        Johnson,
                    )
                    .with_range(4., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(energy_range, err)
                    .with_energy_units(GHz)
            })
            .collect::<Vec<BoundStates>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let bound_dependence = BoundStatesDependence {
            parameters: mag_fields,
            bound_states,
        };

        save_serialize("li2_bound_states", &bound_dependence).unwrap()
    }

    fn field_bound_states() {
        ///////////////////////////////////

        let projection = hi32!(0);
        let energies: Vec<Energy<GHz>> = linspace(-2., -1e-2, 501)
            .iter()
            .map(|x| Energy(x.powi(3), GHz))
            .collect();

        let mag_fields = (0., 1200.);
        let err = 1e-2;

        ///////////////////////////////////

        let start = Instant::now();
        let bound_states = energies
            .par_iter()
            .progress()
            .map(|&energy| {
                let mut particles = Self::get_particles();
                particles.insert(energy.to(Au));

                let problem = |field| {
                    Self::get_problem(projection, field)
                };

                let bound_problem = FieldProblemBuilder::new(&particles, &problem)
                    .with_propagation(
                        LocalWavelengthStepRule::new(1e-4, f64::INFINITY, 500.),
                        Johnson,
                    )
                    .with_range(4., 20., 500.)
                    .build();

                bound_problem
                    .bound_states(mag_fields, err)
            })
            .collect::<Vec<FieldBoundStates>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let bound_dependence = FieldBoundStatesDependence {
            energies: energies.iter().map(|x| x.value()).collect(),
            bound_states,
        };

        save_serialize("li2_field_states", &bound_dependence).unwrap()
    }
}