use std::{
    f64::consts::PI,
    fs::File,
    io::{BufRead, BufReader},
};

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder};
use clebsch_gordan::{half_integer::HalfI32, hu32};
use faer::Mat;

#[allow(unused)]
use indicatif::{ParallelProgressIterator, ProgressIterator};

use quantum::{
    params::{
        Params,
        particle::Particle,
        particle_factory::{RotConst, create_atom},
        particles::Particles,
    },
    units::{
        Au, Unit,
        distance_units::{Angstrom, Distance},
        energy_units::{CmInv, Energy, EnergyUnit, GHz},
        mass_units::{Dalton, Mass},
    },
    utility::{legendre_polynomials, linspace},
};
use rayon::prelude::*;
use scattering_problems::{
    alkali_rotor_atom::{
        AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder, TramBasisRecipe, TramStates,
    },
    potential_interpolation::{PotentialArray, TransitionedPotential, interpolate_potentials},
    rkhs_interpolation::RKHSInterpolation,
    utility::{AnisoHifi, GammaSpinRot},
};
use scattering_solver::potentials::{
    composite_potential::Composite, dispersion_potential::Dispersion, potential::SimplePotential,
};

pub fn get_particles(energy: Energy<impl EnergyUnit>, projection: HalfI32) -> Particles {
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

#[allow(unused)]
pub fn get_problem(
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

#[allow(unused)]
pub fn read_potentials(max_degree: u32) -> [PotentialArray; 2] {
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

pub fn read_extended(max_degree: u32) -> [PotentialArray; 2] {
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

pub fn get_interpolated(
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
