use abm::{HifiProblemBuilder, utility::save_spectrum};
use clebsch_gordan::{half_integer::HalfI32, hi32, hu32};
use quantum::{
    params::{Params, particle::Particle, particle_factory::RotConst},
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
    units::{
        Au,
        energy_units::{CmInv, Energy},
        mass_units::{Dalton, Mass},
    },
    utility::linspace,
};
use scattering_problems::{
    alkali_rotor::{AlkaliRotorProblem, AlkaliRotorProblemBuilder, UncoupledAlkaliRotorProblem},
    alkali_rotor_atom::{ParityBlock, TramBasisRecipe},
    utility::{AnisoHifi, GammaSpinRot},
};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "YbF levels",
    "levels tram" => |_| Self::levels_tram(),
    "levels uncoupled" => |_| Self::levels_uncoupled(),
);

impl Problems {
    fn levels_tram() {
        let projection = hi32!(1);
        let yb_f = get_particle();

        let basis_recipe = TramBasisRecipe {
            l_max: 4,
            n_max: 4,
            n_tot_max: 2,
            parity: ParityBlock::All,
        };

        let alkali_problem = get_problem(projection, &yb_f, &basis_recipe);
        println!("{}", alkali_problem.basis.len());

        let mag_fields = linspace(0., 70., 8);

        let energies: Vec<Vec<f64>> = mag_fields
            .par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels(*mag_field);

                levels
                    .iter()
                    .map(|x| Energy(*x, Au).to(CmInv).value())
                    .collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(
            header,
            &format!("YbF_levels_proj_{}", projection.double_value() / 2),
            &mag_fields,
            &energies,
        )
        .expect("error while saving YbF_Rb levels");
    }

    fn levels_uncoupled() {
        let yb_f = get_particle_uncoupled();
        let alkali_problem = get_problem_uncoupled(&yb_f, 4);
        println!("{}", alkali_problem.basis.len());

        let mag_fields = linspace(0., 70., 70);

        let energies: Vec<Vec<f64>> = mag_fields
            .par_iter()
            .map(|mag_field| {
                let (levels, _) = alkali_problem.levels(*mag_field);

                levels
                    .iter()
                    .map(|x| Energy(*x, Au).to(CmInv).value())
                    .collect()
            })
            .collect();

        let header = "magnetic field [G]\tEnergies [GHz]";
        save_spectrum(
            header,
            &format!("YbF_levels_uncoupled"),
            &mag_fields,
            &energies,
        )
        .expect("error while saving YbF_Rb levels");
    }
}

fn get_particle() -> Particle {
    let mut caf = Particle::new("YbF", Mass(192.9372652, Dalton));

    caf.insert(RotConst(Energy(0.24129, CmInv).to_au()));
    caf.insert(GammaSpinRot(Energy(4.4778e-4, CmInv).to_au()));
    caf.insert(AnisoHifi(Energy(2.84875e-3, CmInv).to_au()));

    caf
}

fn get_problem(
    projection: HalfI32,
    params: &Params,
    basis_recipe: &TramBasisRecipe,
) -> AlkaliRotorProblem {
    let hifi_ybf = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
        .with_hyperfine_coupling(Energy(5.6794e-3, CmInv).to_au())
        .with_total_projection(projection);

    AlkaliRotorProblemBuilder::new(hifi_ybf).build(params, basis_recipe)
}

fn get_particle_uncoupled() -> Particle {
    let mut caf = Particle::new("YbF", Mass(192.9372652, Dalton));

    caf.insert(RotConst(Energy(0.24129, CmInv).to_au()));
    caf.insert(GammaSpinRot(Energy(4.4778e-4, CmInv).to_au()));
    caf.insert(AnisoHifi(Energy(2.84875e-3, CmInv).to_au()));

    caf
}

fn get_problem_uncoupled(particle: &Particle, n_max: u32) -> UncoupledAlkaliRotorProblem {
    let hifi_ybf = HifiProblemBuilder::new(hu32!(1 / 2), hu32!(1 / 2))
        .with_hyperfine_coupling(Energy(5.6794e-3, CmInv).to_au());

    AlkaliRotorProblemBuilder::new(hifi_ybf).build_uncoupled(particle, n_max)
}
