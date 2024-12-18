use std::{f64::consts::PI, fs::File, io::{BufRead, BufReader}};

use faer::Mat;
use quantum::{problem_selector::{get_args, ProblemSelector}, problems_impl, units::{distance_units::{Angstrom, Distance}, energy_units::{CmInv, Energy}}, utility::{legendre_polynomials, linspace}};
use scattering_problems::potential_interpolation::{interpolate_potentials, PotentialArray, TransitionedPotential};
use scattering_solver::{potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::SimplePotential}, utility::save_data};

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "potentials" => |_| Self::potentials(),
    "single elastic cross section" => |_| Self::single_elastic_cross_section(),
    "elastic cross section" => |_| Self::elastic_cross_section(),
);

impl Problems {
    fn potentials() {
        let distances = linspace(7., 20., 200);

        let [pot_array_singlet, pot_array_triplet] = read_potentials(25);
        let mut data = vec![pot_array_triplet.distances.clone()];
        for (_, p) in &pot_array_triplet.potentials {
            data.push(p.clone());
        }

        save_data("SrF_Rb_triplet_dec", "distances\tpotential_decomposition", &data)
            .unwrap();

        let interpolated = get_interpolated(&pot_array_triplet);
        let mut data = vec![distances.clone()];
        for (_, p) in &interpolated {
            let values = distances.iter()
                .map(|&x| p.value(x))
                .collect();

            data.push(values);
        }

        save_data("SrF_Rb_triplet_dec_interpolated", "distances\tpotential_decomposition", &data)
            .unwrap();

        let mut data = vec![pot_array_singlet.distances.clone()];
        for (_, p) in &pot_array_singlet.potentials {
            data.push(p.clone());
        }

        save_data("SrF_Rb_singlet_dec", "distances\tpotential_decomposition", &data)
            .unwrap();

        let interpolated = get_interpolated(&pot_array_singlet);
        let mut data = vec![distances.clone()];
        for (_, p) in &interpolated {
            let values = distances.iter()
                .map(|&x| p.value(x))
                .collect();

            data.push(values);
        }

        save_data("SrF_Rb_singlet_dec_interpolated", "distances\tpotential_decomposition", &data)
            .unwrap();
    }

    fn single_elastic_cross_section() {

    }

    fn elastic_cross_section() {

    }
}

fn read_potentials(max_degree: u32) -> [PotentialArray; 2] {
    let filename = "Rb_SrF/pot.data.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f = File::open(&path).expect(&format!("couldn't find potential in provided path {path:?}"));
    let f = BufReader::new(f);

    let filename = "Rb_SrF/casscf_ex.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f2 = File::open(&path).expect(&format!("couldn't find potential in provided path {path:?}"));
    let f2 = BufReader::new(f2);

    let angle_count = 1 + 180 / 5;
    let r_count = 30;
    let mut values_singlet = Mat::zeros(angle_count, r_count);
    let mut values_triplet = Mat::zeros(angle_count, r_count);
    let mut distances = vec![0.; r_count];
    let angles: Vec<f64> = (0..=180).step_by(5)
        .map(|x| x as f64 / 180. * PI)
        .collect();

    for ((i, line_singlet), line_diff) in f.lines().skip(1).enumerate().zip(f2.lines().skip(1)) {
        let line_singlet = line_singlet.unwrap();
        let splitted_singlet: Vec<&str> = line_singlet.trim().split_whitespace().collect();

        let r: f64 = splitted_singlet[0].parse().unwrap();
        let value: f64 = splitted_singlet[1].parse().unwrap();

        let angle_index = i / r_count;
        let r_index = i % r_count;

        if angle_index > 0 {
            assert!(distances[r_index] == Distance(r, Angstrom).to_au())
        }

        let line_diff = line_diff.unwrap();
        let splitted_diff: Vec<&str> = line_diff.trim().split_whitespace().collect();

        let r_diff: f64 = splitted_diff[0].parse().unwrap();
        let value_diff: f64 = splitted_diff[1].parse().unwrap();
        print!("{value_diff}, ");

        assert!(r_diff == r);

        distances[r_index] = Distance(r, Angstrom).to_au();
        values_singlet[(angle_index, r_index)] = Energy(value, CmInv).to_au();
        values_triplet[(angle_index, r_index)] = Energy(value + value_diff, CmInv).to_au();
    }

    let filename = "weights.txt";
    path.pop();
    path.push(filename);
    let f = File::open(&path).expect(&format!("couldn't find potential in provided path {path:?}"));
    let mut f = BufReader::new(f);

    let mut weights = String::new();
    f.read_line(&mut weights).unwrap();
    let weights: Vec<f64> = weights.trim().split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    let mut potentials_singlet = Vec::new();
    let mut potentials_triplet = Vec::new();
    let polynomials: Vec<Vec<f64>> = angles.iter()
        .map(|x| legendre_polynomials(max_degree, x.cos()))
        .collect();

    for lambda in 0..=max_degree {
        let mut lambda_values = Vec::new();
        for values_col in values_triplet.col_iter() {
            let value: f64 = weights.iter()
                .zip(values_col.iter())
                .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                .map(|((w, v), p)| (lambda as f64 + 0.5) * w * v * p)
                .sum();
            lambda_values.push(value)
        }
        potentials_triplet.push((lambda as u32, lambda_values));

        let mut lambda_values = Vec::new();
        for values_col in values_singlet.col_iter() {
            let value: f64 = weights.iter()
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
        distances: distances.clone()
    };

    let triplets = PotentialArray {
        potentials: potentials_triplet,
        distances
    };

    [singlets, triplets]
}

fn get_interpolated(pot_array: &PotentialArray) -> Vec<(u32, impl SimplePotential)> {
    let interp_potentials = interpolate_potentials(pot_array, 3);
    
    let mut potentials_far = Vec::new();
    for _ in &interp_potentials {
        potentials_far.push(Composite::new(Dispersion::new(0., 0)));
    }

    potentials_far[0].add_potential(Dispersion::new(-3495.30040855597, -6))
        .add_potential(Dispersion::new(-516911.950541056, -8));
    potentials_far[1].add_potential(Dispersion::new(17274.8363457991, -7))
        .add_potential(Dispersion::new(4768422.32042577, -9));
    potentials_far[2].add_potential(Dispersion::new(-288.339392609436, -6))
        .add_potential(Dispersion::new(-341345.136436851, -8));
    potentials_far[3].add_potential(Dispersion::new(-12287.2175217778, -7))
        .add_potential(Dispersion::new(-1015530.4019772, -9));
    potentials_far[4].add_potential(Dispersion::new(-51933.9885816, -8))
        .add_potential(Dispersion::new(-3746260.46991, -10));

    let transition = |r| {
        let r_min = Distance(9., Angstrom).to_au();
        let r_max = Distance(10., Angstrom).to_au();

        if r <= r_min {
            1.
        } else if r >= r_max {
            0.
        } else {
            0.5 * (1. + f64::cos(PI * (r - r_min) / (r_max - r_min)))
        }
    };

    interp_potentials.into_iter()
        .zip(potentials_far.into_iter())
        .map(|((lambda, near), far)| {
            let combined = TransitionedPotential::new(near, far, transition);

            (lambda, combined)
        })
        .collect()
}