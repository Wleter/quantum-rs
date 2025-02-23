use std::{f64::consts::PI, fs::File, io::{BufRead, BufReader}, time::Instant};

use faer::Mat;
use gauss_quad::GaussLegendre;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use quantum::{params::{particle::Particle, particle_factory::{create_atom, RotConst}, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{distance_units::Angstrom, energy_units::{CmInv, Energy, Kelvin}, mass_units::{Dalton, Mass}, Au, Unit}, utility::{legendre_polynomials, linspace, logspace}};
use scattering_problems::{potential_interpolation::{interpolate_potentials, PotentialArray, TransitionedPotential}, rotor_atom::{RotorAtomBasisDescription, RotorAtomBasisElement, RotorAtomProblemBuilder}, utility::{RotorJMax, RotorJTot, RotorLMax}, BasisDescription, ScatteringProblem};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::MultiRatioNumerov, propagator::MultiStepRule}, potentials::{dispersion_potential::Dispersion, potential::{MatPotential, Potential, SimplePotential}}, utility::save_data};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Spinless AlF + Rb",
    "potential values" => |_| Self::potential_values(),
    "elastic cross section 1 channel" => |_| Self::elastic_cross_section_1chan(),
    "elastic cross section" => |_| Self::elastic_cross_section(),
    "inelastic cross section" => |_| Self::inelastic_cross_section(),
    "inelastic cross section j_tot" => |_| Self::inelastic_cross_section_j_tot()
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

        let interp_potentials = interpolate_potentials(&mut pot_array, 3);
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
        let pot_array = read_potential();

        let entrance = RotorAtomBasisElement {
            l: 0,
            j: 0,
        };

        ///////////////////////////////
        
        let start = Instant::now();

        let mut particles = get_particles(is_rb_87, energy_relative.to(Au));
        particles.insert(RotorLMax(80));
        particles.insert(RotorJMax(80));

        let scattering_problem = get_potentials(&pot_array, &particles);
        let potential = scattering_problem.potential;
        let mut asymptotic = scattering_problem.asymptotic;
        asymptotic.entrance = scattering_problem.basis_description.index_for(&entrance);
        particles.insert(asymptotic);

        let id = Mat::<f64>::identity(potential.size(), potential.size());
        let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
        let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
        let mut numerov = MultiRatioNumerov::new(&potential, &particles, step_rule, boundary);

        numerov.propagate_to(200.);
        let cross_section = numerov.data.calculate_s_matrix().get_elastic_cross_sect();
        
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
        let energy = Energy(1e-7, Kelvin);
        let rot_max = 80;
        let pot_array = read_potential();

        let entrance = RotorAtomBasisElement {
            l: 0,
            j: 0,
        };

        ///////////////////////////////

        for is_rb_87 in [true, false] {
            let start = Instant::now();
            let rot_maxes: Vec<u32> = (0..=rot_max).collect();
    
            let cross_sections: Vec<f64> = rot_maxes.par_iter()
                .progress()
                .map(|x| {
                    let mut particles = get_particles(is_rb_87, energy.to(Au));
            
                    particles.insert(RotorLMax(*x));
                    particles.insert(RotorJMax(*x));
    
                    let scattering_problem = get_potentials(&pot_array, &particles);
                    let potential = scattering_problem.potential;
                    let mut asymptotic = scattering_problem.asymptotic;
                    asymptotic.entrance = scattering_problem.basis_description.index_for(&entrance);
                    particles.insert(asymptotic);
        
                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
                    let mut numerov = MultiRatioNumerov::new(&potential, &particles, step_rule, boundary);
        
                    numerov.propagate_to(200.);
                    let cross_section = numerov.data.calculate_s_matrix().get_elastic_cross_sect();
    
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

    fn inelastic_cross_section() {
        let energies = logspace(-7., 0., 500);

        ///////////////////////////////

        let entrance = RotorAtomBasisElement {
            l: 1,
            j: 1,
        };

        let pot_array = read_potential();

        for is_rb_87 in [true, false] {
            let start = Instant::now();
    
            let cross_sections: Vec<[f64; 5]> = energies.par_iter()
                .progress()
                .map(|x| {
                    let energy = Energy(*x, Kelvin);

                    let mut particles = get_particles(is_rb_87, energy.to(Au));
            
                    particles.insert(RotorLMax(64));
                    particles.insert(RotorJMax(64));
    
                    let scattering_problem = get_potentials(&pot_array, &particles);
                    let potential = scattering_problem.potential;
                    let mut asymptotic = scattering_problem.asymptotic;
                    asymptotic.entrance = scattering_problem.basis_description.index_for(&entrance);
                    particles.insert(asymptotic);
        
                    let id = Mat::<f64>::identity(potential.size(), potential.size());
                    let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
                    let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
                    let mut numerov = MultiRatioNumerov::new(&potential, &particles, step_rule, boundary);
        
                    numerov.propagate_to(200.);
                    let s_matrix = numerov.data.calculate_s_matrix();

                    let inel_cross_section = s_matrix.get_inelastic_cross_sect_to(0) / Angstrom::TO_AU_MUL.powi(2);
                    let inel_tot_cross_section = s_matrix.get_inelastic_cross_sect() / Angstrom::TO_AU_MUL.powi(2);
                    let el_cross_section = s_matrix.get_elastic_cross_sect() / Angstrom::TO_AU_MUL.powi(2);
                    let el_s_length_re = s_matrix.get_scattering_length().re;
                    let el_s_length_im = s_matrix.get_scattering_length().im;
    
                    [inel_cross_section, inel_tot_cross_section, el_cross_section, el_s_length_re, el_s_length_im]
                })
                .collect();
    
            let elapsed = start.elapsed();
            println!("calculated in {}", elapsed.hhmmssxxx());
    
            let inelastic = cross_sections.iter().map(|x| x[0]).collect();
            let inelastic_tot = cross_sections.iter().map(|x| x[1]).collect();
            let elastic = cross_sections.iter().map(|x| x[2]).collect();
            let s_re = cross_sections.iter().map(|x| x[3]).collect();
            let s_im = cross_sections.iter().map(|x| x[4]).collect();

            let header = "energies\tcross_sections";
            let data = vec![energies.clone(), inelastic, inelastic_tot, elastic, s_re, s_im];
    
            save_data(&format!("AlF_Rb_inelastic_section_{is_rb_87}"), header, &data)
                .unwrap()
        }
    }

    fn inelastic_cross_section_j_tot() {
        let energies = logspace(-7., 0., 500);
        let j_tot_max = 5;

        let entrance = RotorAtomBasisElement {
            l: 1,
            j: 1,
        };

        ///////////////////////////////

        let pot_array = read_potential();
        
        for is_rb_87 in [true, false] {
            let mut data = vec![energies.clone()];

            for j_tot in 0..=j_tot_max {
                let start = Instant::now();
        
                let cross_sections: Vec<f64> = energies.par_iter()
                    .progress()
                    .map(|x| {
                        let energy = Energy(*x, Kelvin);

                        let mut particles = get_particles(is_rb_87, energy.to(Au));
                
                        particles.insert(RotorLMax(64));
                        particles.insert(RotorJMax(64));
                        particles.insert(RotorJTot(j_tot));
        
                        let scattering_problem = get_potentials(&pot_array, &particles);
                        let potential = scattering_problem.potential;
                        let mut asymptotic = scattering_problem.asymptotic;
                        asymptotic.entrance = scattering_problem.basis_description.index_for(&entrance);
                        particles.insert(asymptotic);
            
                        let id = Mat::<f64>::identity(potential.size(), potential.size());
                        let boundary = Boundary::new(5., Direction::Outwards, (1.001 * &id, 1.002 * &id));
                        let step_rule = MultiStepRule::new(1e-4, f64::INFINITY, 500.);
                        let mut numerov = MultiRatioNumerov::new(&potential, &particles, step_rule, boundary);
            
                        numerov.propagate_to(200.);
                        let s_matrix = numerov.data.calculate_s_matrix();

                        let inel_cross_section = s_matrix.get_inelastic_cross_sect() / Angstrom::TO_AU_MUL.powi(2);
        
                        inel_cross_section
                    })
                    .collect();

                let elapsed = start.elapsed();
                println!("calculated in {}", elapsed.hhmmssxxx());
                
                data.push(cross_sections);
            }

            let header = "energies\tcross_sections";
    
            save_data(&format!("AlF_Rb_inelastic_section_{is_rb_87}_j_tots"), header, &data)
                .unwrap()
        }
    }
}

fn get_potentials(pot_array: &PotentialArray, particles: &Particles) -> ScatteringProblem<impl MatPotential + use<>, RotorAtomBasisDescription> {
    let interp_potentials = interpolate_potentials(pot_array, 3);
    
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
            let combined = TransitionedPotential::new(near, far, transition);

            (lambda, combined)
        })
        .collect();

    RotorAtomProblemBuilder::new(potentials)
        .build(particles)
}

fn get_particles(is_rb_87: bool , energy: Energy<Au>) -> Particles {
    let rb = if is_rb_87 { create_atom("Rb87").unwrap() } else { create_atom("Rb85").unwrap() };
    let alf = Particle::new("AlF", Mass(27. + 19., Dalton));

    let mut particles = Particles::new_pair(rb, alf, energy);

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

    PotentialArray::new(distances, potentials)
}
