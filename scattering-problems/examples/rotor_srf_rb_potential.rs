use std::{fs::File, io::{self, BufRead, BufReader, Write}};

use clebsch_gordan::hi32;
use faer::{Col, Mat};
use quantum::{problem_selector::{get_args, ProblemSelector}, problems_impl, units::{Energy, Kelvin}, utility::linspace};

mod common;
use common::{PotentialType, ScalingType, Scalings, srf_rb_functionality::*};
use regex::Regex;
use scattering_problems::rotor_atom::{RotorAtomBasisRecipe, RotorAtomProblemBuilder};
use scattering_solver::{potentials::potential::Potential, utility::save_spectrum};

use crate::common::potential_cutting::PotentialCutting;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb potentials",
    "manipulate potential" => |_| Self::manipulate_potential(),
    "cut potential" => |_| Self::cut_potential(),
    "transform wavefunction adiabatically" => |_| Self::wave_adiabats(),
);

impl Problems {
    // todo! very weirdly done
    fn manipulate_potential() -> ! {
        
        let mut potential_type = PotentialType::Singlet;
        let mut n_max = 10;
        
        loop {
            let mut buffer = String::new();
            io::stdin().read_line(&mut buffer).unwrap();

            let configs = buffer.trim().split(",").map(|x|
                if x == "singlet" {
                    Config::Potential(PotentialType::Singlet)
                } else if x == "triplet" {
                    Config::Potential(PotentialType::Triplet)
                } else if x.starts_with("n_max=") {
                    Config::NMax(x[6..x.len()].parse().unwrap())
                } else {
                    let split: Vec<_> = x.split(" ").collect();

                    Config::Scaling(split[0].parse().unwrap(), split[1].parse().unwrap())
                }
            );
            let mut scalings = Scalings::default();

            for c in configs {
                match c {
                    Config::Potential(pes_type) => potential_type = pes_type,
                    Config::NMax(n) => n_max = n,
                    Config::Scaling(lambda, scaling) => {
                        scalings.scalings.push(scaling);
                        match lambda {
                            -1 => {
                                scalings.scaling_types.push(ScalingType::Anisotropic)
                            },
                            -2 => {
                                scalings.scaling_types.push(ScalingType::Full)
                            },
                            0..=25 => {
                                scalings.scaling_types.push(ScalingType::Legendre(lambda as u32))
                            },
                            _ => panic!()
                        }
                    },
                }
            }
    
            single_manipulate_potential(potential_type, scalings, n_max);
        }
    }

    // todo! very wrong approach probably so far, there is 0 guarantee the basis is effective TRAM, 
    // only adiabats are correct (but maybe QR gives good estimate?)
    fn cut_potential() {
        let pes_type = PotentialType::Singlet; 
        let n_max = 50;
        let cut_n_max = 10;

        let basis_recipe = RotorAtomBasisRecipe {
            l_max: n_max,
            n_max: n_max,
            ..Default::default()
        };

        let distances = linspace(5., 80., 800);

        let [pot_array_singlet, pot_array_triplet] = read_extended(25);
        let pes = match pes_type {
            PotentialType::Singlet => pot_array_singlet,
            PotentialType::Triplet => pot_array_triplet,
        };

        let interpolated = get_interpolated(&pes);

        let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
        let problem = RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);
        let potential = PotentialCutting::new(problem.potential, cut_n_max + 1);

        let mut data = vec![];
        let mut potential_value = Mat::zeros(potential.size(), potential.size());
        for &r in &distances {
            potential.value_inplace(r, &mut potential_value);

            data.push(
                potential_value
                    .self_adjoint_eigenvalues(faer::Side::Lower)
                    .unwrap(),
            );
        }

        save_spectrum(
            &format!("SrF_Rb_{pes_type}_adiabat_{n_max}_cut_{cut_n_max}"),
            "distance\tadiabat",
            &distances,
            &data,
        )
        .unwrap();
    }

    fn wave_adiabats() {
        let pes_type = PotentialType::Singlet; 
        let n_max = 175;
        let filename = format!("data/wave_function_{pes_type}_{n_max}.output");
        let basis_recipe = RotorAtomBasisRecipe {
            l_max: n_max,
            n_max: n_max,
            ..Default::default()
        };
        let file_wave_adiabat = format!("data/wave_function_{pes_type}_{n_max}_adiabat.output");
        let mut file_adiabat = File::create(&file_wave_adiabat).unwrap();

        let file = File::open(&filename).unwrap();
        let mut rdr = WaveFunctionIterator::new(BufReader::new(file), n_max);

        let [pot_array_singlet, pot_array_triplet] = read_extended(25);
        let pes = match pes_type {
            PotentialType::Singlet => pot_array_singlet,
            PotentialType::Triplet => pot_array_triplet,
        };
        let interpolated = get_interpolated(&pes);
        let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
        let problem = RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);
        let potential = &problem.potential;

        let mut potential_value = Mat::zeros(potential.size(), potential.size());
        let mut vector: Col<f64> = Col::zeros(potential.size());

        while let Some(mut state) = rdr.next_wavefunction() {
            println!("{} {}", state.energy, state.node);
            file_adiabat.write_fmt(format_args!("# WAVEFUNCTION FOR STATE {} AT ENERGY  {}     GHZ  RELATIVE TO REFERENCE ENERGY\n", state.node, state.energy)).unwrap();
            file_adiabat.flush().unwrap();

            for (r, coeffs) in &mut state {
                potential.value_inplace(r, &mut potential_value);
                for (d, &c) in vector.iter_mut().zip(&coeffs) {
                    *d = c
                }
                let eigen = potential_value
                    .self_adjoint_eigen(faer::Side::Lower)
                    .unwrap();

                let result = eigen.U().transpose() * &vector;

                file_adiabat.write_fmt(format_args!("{r:e}")).unwrap();
                for c in result.iter() {
                    file_adiabat.write_fmt(format_args!(" {c:e}")).unwrap();
                }
                file_adiabat.write(b"\n").unwrap();
                file_adiabat.flush().unwrap()
            }
        }
        
        file_adiabat.flush().unwrap();
        println!("Saved in {file_wave_adiabat}");
    }
}

enum Config {
    Potential(PotentialType),
    NMax(u32),
    Scaling(i32, f64)
}

fn single_manipulate_potential(
    pes_type: PotentialType, 
    scalings: Scalings, 
    n_max: u32
) {
    let basis_recipe = RotorAtomBasisRecipe {
            l_max: n_max,
            n_max: n_max,
            ..Default::default()
        };

        let distances = linspace(5., 80., 800);

        let [pot_array_singlet, pot_array_triplet] = read_extended(25);
        let pes = match pes_type {
            PotentialType::Singlet => pot_array_singlet,
            PotentialType::Triplet => pot_array_triplet,
        };

        let interpolated = get_interpolated(&pes);

        let interpolated = scalings.scale(&interpolated);

        let atoms = get_particles(Energy(1e-7, Kelvin), hi32!(0));
        let problem = RotorAtomProblemBuilder::new(interpolated).build(&atoms, &basis_recipe);

        let mut data = vec![];
        let mut potential_value = Mat::zeros(problem.potential.size(), problem.potential.size());
        for &r in &distances {
            problem.potential.value_inplace(r, &mut potential_value);

            data.push(
                potential_value
                    .self_adjoint_eigenvalues(faer::Side::Lower)
                    .unwrap(),
            );
        }

        let pes = match pes_type {
            PotentialType::Singlet => "singlet",
            PotentialType::Triplet => "triplet",
        };

        save_spectrum(
            &format!("SrF_Rb_{pes}_adiabat_scaled"),
            "distance\tadiabat",
            &distances,
            &data,
        )
        .unwrap();
}

#[derive(Debug)]
pub struct WaveFunctionIterator {
    n_max: u32,
    reader: BufReader<File>,
    header_re: Regex,
    peeked: Option<String>,
}

#[derive(Debug)]
pub struct WaveDataIter<'a> {
    node: u32,
    energy: f64,
    parent: &'a mut WaveFunctionIterator,
    finished: bool,
}

impl WaveFunctionIterator {
    pub fn new(reader: BufReader<File>, n_max: u32) -> Self {
        let header_re = Regex::new(
            r"WAVEFUNCTION FOR STATE\s+(\d+)\s+AT ENERGY\s+([-+]?\d*\.\d+(?:[eE][+-]?\d+)?)"
        ).unwrap();
        WaveFunctionIterator { n_max, reader, header_re, peeked: None }
    }

    pub fn next_wavefunction<'a>(&'a mut self) -> Option<WaveDataIter<'a>> {
        let mut line = String::new();

        loop {
            line.clear();
            let n = if let Some(buf) = self.peeked.take() {
                line = buf;
                line.len()
            } else {
                match self.reader.read_line(&mut line) {
                    Ok(0) => return None,
                    Ok(n) => n,
                    Err(e) => panic!("{e}"),
                }
            };
            if n == 0 { return None; }
            if let Some(caps) = self.header_re.captures(&line) {
                let node: u32 = caps[1].parse().unwrap();
                let energy: f64 = caps[2].parse().unwrap();

                let data_wave_iter = WaveDataIter { node, energy, parent: self, finished: false };
                return Some(data_wave_iter);
            }
        }
    }
}

impl<'a> Iterator for WaveDataIter<'a> {
    type Item = (f64, Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut line = String::new();

        loop {
            line.clear();
            match self.parent.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    return None;
                }
                Ok(_) => {
                    let s = line.trim();

                    if self.parent.header_re.is_match(s) {
                        self.parent.peeked = Some(line.clone());
                        self.finished = true;
                        return None;
                    }
                    if s.is_empty() || s.starts_with('#') {
                        continue;
                    }
                    break;
                }
                Err(e) => panic!("{e}"),
            }
        }

        let mut parts = line.split_whitespace();
        let r: f64 = parts.next().unwrap().parse().unwrap();
        let mut coeffs: Vec<f64> = parts.map(|tok| tok.parse().unwrap()).collect();

        while coeffs.len() < (self.parent.n_max + 1) as usize {
            line.clear();
            match self.parent.reader.read_line(&mut line) {
                Ok(0) => {
                    panic!("EOF before reading all coefs");
                }
                Ok(_) => {
                    let s = line.trim();
                    if s.is_empty() || s.starts_with('#') {
                        continue;
                    }
                    if self.parent.header_re.is_match(s) {
                        panic!("New wavefunction before reading all coefs")
                    }

                    coeffs.extend(
                        s.split_whitespace()
                            .map(|tok| tok.parse::<f64>().unwrap())
                    );
                }
                Err(e) => panic!("{e}"),
            }
        }
        assert_eq!(coeffs.len() as u32, self.parent.n_max + 1);

        Some((r, coeffs))
    }
}