use std::{f64::consts::PI, fs::File, io::{BufRead, BufReader}, mem::swap, time::Instant};

use faer::Mat;
use gauss_quad::GaussLegendre;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use quantum::{params::{particle::Particle, particle_factory::{create_atom, RotConst}, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{distance_units::Angstrom, energy_units::{CmInv, Energy, Kelvin}, mass_units::{Dalton, Mass}, Au, Unit}, utility::{legendre_polynomials, linspace}};
use rusty_fitpack::{splev, splrep};
use scattering_problems::{rotor_atom::RotorAtomProblemBuilder, utility::{RotorJMax, RotorJTot, RotorLMax}};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::faer_backed::FaerRatioNumerov, propagator::MultiStepRule}, observables::s_matrix::HasSMatrix, potentials::{dispersion_potential::Dispersion, potential::{Potential, SimplePotential, SubPotential}}, utility::save_data};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Spinless AlF + Rb",
    "potential values" => |_| Self::potential_values(),
    "elastic cross section" => |_| Self::elastic_cross_section_1chan(),
    "elastic cross section" => |_| Self::elastic_cross_section()
);

impl Problems {
    fn potential_values() {
        let mut pot_array = read_potential();
        let mut data = vec![pot_array.distances.clone()];
        for (_, p) in &pot_array.potentials {
            data.push(p.clone());
        }

        save_data("potential_dec_AlF_Rb", "", &data)
            .unwrap();

        let interp_potentials = interpolate_potentials(&mut pot_array, 0.1);
        let distances = linspace(4., 50., 1000);

        let mut data = vec![distances.clone()];
        for (_, p) in &interp_potentials {
            let values = distances.iter()
                .map(|&x| Energy(p.value(x), Au).to(CmInv).value())
                .collect();
            data.push(values);
        }

        save_data("interpolated_dec_AlF_Rb", "", &data)
            .unwrap();
    }

    fn elastic_cross_section_1chan() {
        let is_rb_87 = true;
        let energy_relative = Energy(1e-7, Kelvin);
        let channel = 0;

        ///////////////////////////////
        
        let start = Instant::now();

        let mut particles = get_particles(is_rb_87, energy_relative.to(Au));
        particles.insert(RotorLMax(80));
        particles.insert(RotorJMax(80));

        let mut pot_array = read_potential();
        let potential = get_potentials(&mut pot_array, &particles);

        let id = Mat::<f64>::identity(potential.size(), potential.size());
        let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
        let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
        let mut numerov = FaerRatioNumerov::new(&potential, &particles, step_rule, boundary);

        numerov.propagate_to(200.);
        let cross_section = numerov.data.calculate_s_matrix(channel).get_elastic_cross_sect();
        
        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());
        if is_rb_87 {
            print!("87Rb + AlF: ")
        } else {
            print!("85Rb + AlF: ")
        }
        println!("{}", cross_section / Angstrom::TO_AU_MUL.powi(2));
    }

    fn elastic_cross_section() {
        let channel = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        let rot_max = 80;

        ///////////////////////////////

        for is_rb_87 in [true, false] {
            let start = Instant::now();
            let rot_maxes: Vec<u32> = (0..=rot_max).collect();
    
            let cross_sections: Vec<f64> = rot_maxes.par_iter()
                .progress()
                .map(|x| {
                    let mut particles = get_particles(is_rb_87, energy_relative.to(Au));
            
                    particles.insert(RotorLMax(*x));
                    particles.insert(RotorJMax(*x));
    
                    let mut pot_array = read_potential();
                    let potential = get_potentials(&mut pot_array, &particles);
        
                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
                    let mut numerov = FaerRatioNumerov::new(&potential, &particles, step_rule, boundary);
        
                    numerov.propagate_to(200.);
                    let cross_section = numerov.data.calculate_s_matrix(channel).get_elastic_cross_sect();
    
                    cross_section / Angstrom::TO_AU_MUL.powi(2)
                })
                .collect();
    
            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());
            let rot_maxes = (0..=rot_max).map(|x| x as f64).collect();
    
            let header = "mag_field\telastic_cross_section";
            let data = vec![rot_maxes, cross_sections];
    
            save_data(&format!("AlF_Rb_elastic_section_{is_rb_87}"), header, &data)
                .unwrap()
        }
    }
}

fn get_potentials<'a>(pot_array: &'a mut PotentialArray, particles: &Particles) -> impl Potential<Space = Mat<f64>> + 'a {
    let interp_potentials = interpolate_potentials(pot_array, 0.001);
    
    let mut potentials_far = Vec::new();
    for _ in &interp_potentials {
        potentials_far.push(Dispersion::new(0., 0));
    }
    potentials_far[0] = Dispersion::new(-1096.4, -6);
    potentials_far[2] = Dispersion::new(-73.8, -6);

    let transition = |r| {
        if r <= 40. {
            1.
        } else if r >= 50. {
            0.
        } else {
            0.5 * (1. + f64::cos(PI * (r - 40.) / 10.))
        }
    };

    let potentials = interp_potentials.into_iter()
        .zip(potentials_far.into_iter())
        .map(|((lambda, near), far)| {
            let combined = TransitionedPotential {
                near,
                far,
                transition,
            };

            (lambda, combined)
        })
        .collect();

    RotorAtomProblemBuilder::new(potentials)
        .build_space_fixed(particles)
}

fn get_particles(is_rb_87: bool , energy: Energy<Au>) -> Particles {
    let rb = if is_rb_87 { create_atom("Rb87").unwrap() } else { create_atom("Rb85").unwrap() };
    let srf = Particle::new("AlF", Mass(27. + 19., Dalton));

    let mut particles = Particles::new_pair(rb, srf, energy);

    let mass = if is_rb_87 {
        30.0707761438229
    } else {
        29.8280043494141
    };
    particles.insert(Mass(mass, Dalton).to(Au));

    particles.insert(RotorJTot(0));
    particles.insert(RotConst(Energy(0.549992, CmInv).to_au()));

    particles
}

fn read_potential() -> PotentialArray {
    let filename = "X_2Ap_Rb+AlF";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("txt");
    let f = File::open(&path).expect(&format!("couldn't find potential in provided path {path:?}"));
    let f = BufReader::new(f);

    let angle_count = 11;
    let r_count = 56;
    let mut values = Mat::zeros(angle_count, r_count);
    let mut distances = vec![0.; r_count];
    let mut angles = vec![0.; angle_count];
    for (i, line) in f.lines().skip(1).enumerate() {
        let line = line.unwrap();
        let splitted: Vec<&str> = line.trim().split_whitespace().collect();

        let r: f64 = splitted[0].parse().unwrap();
        let angle: f64 = splitted[1].parse().unwrap();
        let angle = angle * PI / 180.;
        let value: f64 = splitted[2].parse().unwrap();

        let angle_index = i / r_count;
        let r_index = i % r_count;

        if angle_index > 0 {
            assert!(distances[r_index] == r)
        }
        if r_index > 0 {
            assert!(angles[angle_index] == angle)
        }

        distances[r_index] = r;
        angles[angle_index] = angle;
        values[(angle_index, r_index)] = value
    }

    let quad = GaussLegendre::new(angle_count).unwrap();
    let angles_quad: Vec<f64> = quad.nodes().map(|x| x.acos()).collect();
    for (angle, q_angle) in angles.iter().zip(&angles_quad) {
        assert!((angle / q_angle - 1.).abs() < 1e-5)
    }

    let mut potentials = Vec::new();
    let polynomials: Vec<Vec<f64>> = quad.nodes()
        .map(|x| legendre_polynomials(angle_count as u32 - 1, *x))
        .collect();

    for lambda in 0..=(angle_count - 1) {
        let mut lambda_values = Vec::new();
        for values_col in values.col_iter() {
            let value: f64 = quad.weights()
                .zip(values_col.iter())
                .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                .map(|((w, v), p)| (lambda as f64 + 0.5) * w * v * p)
                .sum();
            lambda_values.push(Energy(value, CmInv).to_au())
        }
        potentials.push((lambda as u32, lambda_values));
    }

    PotentialArray {
        potentials,
        distances
    }
}

#[derive(Debug)]
struct PotentialArray {
    potentials: Vec<(u32, Vec<f64>)>,
    distances: Vec<f64>,
}

fn interpolate_potentials<'a>(pot_array: &'a mut PotentialArray, res: f64) -> Vec<(u32, InterpolatedPotential<'a>)> {
    let r_min = 4.;
    let r_max = 50.;
    let r_no = (45. / res).ceil() as usize;
    let r_step = (r_max - r_min) / r_no as f64;

    let x = linspace(4., 50., (45. / res).ceil() as usize);
    let mut interpolated = Vec::new();

    for (lambda, potential) in &mut pot_array.potentials {
        let (t, c, k) = splrep(pot_array.distances.clone(), potential.clone(), None, None, None, None, None, None, None, None, None, None);
        swap(potential, &mut splev(t, c, k, x.clone(), 0));

        let interp = InterpolatedPotential {
            values: potential,
            r_min,
            r_max,
            r_step,
        };

        interpolated.push((*lambda, interp))
    }

    interpolated
}

#[derive(Clone)]
pub struct InterpolatedPotential<'a> {
    values: &'a [f64],
    r_min: f64,
    r_max: f64,
    r_step: f64,
}

impl<'a> InterpolatedPotential<'a> {
    fn value_internal(&self, r: f64) -> f64 {
        if self.r_min >= r {
            return self.values[0]
        }
        if self.r_max <= r {
            return *self.values.last().unwrap()
        }

        let mut progress = (r - self.r_min).max(0.) / self.r_step;

        let index = progress.floor();
        progress -= index;
        let index = index as usize;
        if index == self.values.len() - 1 {
            return self.values[index]
        }

        self.values[index] + progress * (self.values[index + 1] - self.values[index])
    }
}

impl Potential for InterpolatedPotential<'_> {
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        *value = self.value_internal(r);
    }

    fn size(&self) -> usize {
        1
    }
}

impl SubPotential for InterpolatedPotential<'_> {
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        *value += self.value_internal(r)
    }
}

pub struct TransitionedPotential<P, V, F>
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    near: P,
    far: V,
    transition: F
}

impl<P, V, F> Potential for TransitionedPotential<P, V, F> 
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    type Space = f64;

    fn value_inplace(&self, r: f64, value: &mut Self::Space) {
        let a = (self.transition)(r);
        assert!(a >= 0. && a <= 1.);

        if a == 0. {
            self.far.value_inplace(r, value);
        } else if a == 1. {
            self.near.value_inplace(r, value);
        } else {
            *value = a * self.near.value(r) + (1. - a) * self.far.value(r)
        }
    }

    fn size(&self) -> usize {
        1
    }
}

impl<P, V, F> SubPotential for TransitionedPotential<P, V, F> 
where
    P: SimplePotential,
    V: SimplePotential,
    F: Fn(f64) -> f64
{
    fn value_add(&self, r: f64, value: &mut Self::Space) {
        let a = (self.transition)(r);
        assert!(a >= 0. && a <= 1.);

        if a == 0. {
            *value += self.far.value(r);
        } else if a == 1. {
            *value += self.near.value(r);
        } else {
            *value += a * self.near.value(r) + (1. - a) * self.far.value(r)
        }
    }
}