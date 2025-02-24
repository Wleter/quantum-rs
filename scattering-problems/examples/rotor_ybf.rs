use abm::{utility::save_spectrum, HifiProblemBuilder};
use clebsch_gordan::{hi32, half_integer::HalfI32, hu32};
use quantum::{params::{particle::Particle, particle_factory::RotConst}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{energy_units::{CmInv, Energy}, mass_units::{Dalton, Mass}, Au}, utility::linspace};
use scattering_problems::{alkali_rotor::{AlkaliRotorProblem, AlkaliRotorProblemBuilder, UncoupledAlkaliRotorProblem}, utility::{AnisoHifi, GammaSpinRot, RotorJMax, RotorJTotMax, RotorLMax}};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "YbF levels",
    "levels" => |_| Self::levels(),
    "levels uncoupled" => |_| Self::levels_uncoupled(),
);

impl Problems {
    fn levels() {
        let projection = hi32!(1);
        let yb_f = get_particle();
        let alkali_problem = get_problem(projection, &yb_f);
        println!("{}", alkali_problem.basis.len());

        let mag_fields = linspace(0., 70., 8);

        let energies: Vec<Vec<f64>> = mag_fields.par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels_at_field(*mag_field);

                levels.iter().map(|x| Energy(*x, Au).to(CmInv).value()).collect()
            })
            .collect();
        
        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, &format!("YbF_levels_proj_{}", projection.double_value() / 2), &mag_fields, &energies)
            .expect("error while saving YbF_Rb levels");
    }

    fn levels_uncoupled() {
        let yb_f = get_particle_uncoupled();
        let alkali_problem = get_problem_uncoupled(&yb_f);
        println!("{}", alkali_problem.basis.len());

        let mag_fields = linspace(0., 70., 70);

        let energies: Vec<Vec<f64>> = mag_fields.par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels_at_field(*mag_field);

                levels.iter().map(|x| Energy(*x, Au).to(CmInv).value()).collect()
            })
            .collect();
        
        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(header, &format!("YbF_levels_uncoupled"), &mag_fields, &energies)
            .expect("error while saving YbF_Rb levels");
    }
}

fn get_particle() -> Particle {
    let mut caf = Particle::new("YbF", Mass(192.9372652, Dalton));

    caf.insert(RotorJMax(4));
    caf.insert(RotorLMax(4));
    caf.insert(RotorJTotMax(2));
    caf.insert(RotConst(Energy(0.24129, CmInv).to_au()));
    caf.insert(GammaSpinRot(Energy(4.4778e-4, CmInv).to_au()));
    caf.insert(AnisoHifi(Energy(2.84875e-3, CmInv).to_au()));
    
    caf
}

fn get_problem(projection: HalfI32, particle: &Particle) -> AlkaliRotorProblem {
    let hifi_ybf = HifiProblemBuilder::new(hu32!(1/2), hu32!(1/2))
        .with_hyperfine_coupling(Energy(5.6794e-3, CmInv).to_au())
        .with_total_projection(projection);

    AlkaliRotorProblemBuilder::new(hifi_ybf)
        .build(particle)
}

fn get_particle_uncoupled() -> Particle {
    let mut caf = Particle::new("YbF", Mass(192.9372652, Dalton));

    caf.insert(RotorJMax(4));
    caf.insert(RotorLMax(4));
    caf.insert(RotConst(Energy(0.24129, CmInv).to_au()));
    caf.insert(GammaSpinRot(Energy(4.4778e-4, CmInv).to_au()));
    caf.insert(AnisoHifi(Energy(2.84875e-3, CmInv).to_au()));
    
    caf
}

fn get_problem_uncoupled(particle: &Particle) -> UncoupledAlkaliRotorProblem {
    let hifi_ybf = HifiProblemBuilder::new(hu32!(1/2), hu32!(1/2))
        .with_hyperfine_coupling(Energy(5.6794e-3, CmInv).to_au());

    AlkaliRotorProblemBuilder::new(hifi_ybf)
        .build_uncoupled(particle)
}
