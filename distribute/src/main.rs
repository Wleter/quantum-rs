use std::{
    f64::consts::PI,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use quantum::{
    params::{
        Params,
        particle::Particle,
        particle_factory::{RotConst, create_atom},
        particles::Particles,
    },
    states::spins::clebsch_gordan::{half_integer::HalfI32, hi32, hu32},
    units::{
        Au, Unit,
        distance_units::{Angstrom, Distance},
        energy_units::{CmInv, Energy, EnergyUnit, GHz, Kelvin},
        mass_units::{Dalton, Mass},
    },
    utility::{legendre_polynomials, linspace},
};
use scattering_problems::{
    FieldScatteringProblem,
    abm::{DoubleHifiProblemBuilder, HifiProblemBuilder},
    alkali_rotor_atom::{
        AlkaliRotorAtomProblem, AlkaliRotorAtomProblemBuilder, TramBasisRecipe, TramStates,
    },
    potential_interpolation::{PotentialArray, TransitionedPotential, interpolate_potentials},
    rkhs_interpolation::RKHSInterpolation,
    utility::{AnisoHifi, GammaSpinRot},
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    faer::Mat,
    numerovs::{LocalWavelengthStepRule, multi_numerov::MultiRNumerov},
    potentials::{
        composite_potential::Composite,
        dispersion_potential::Dispersion,
        potential::{Potential, SimplePotential},
    },
    propagator::{CoupledEquation, Propagator},
    utility::save_data,
};
use timely_hpc::{distribute, timely};

fn main() {
    let entrance = 0;

    let projection = hi32!(1);
    let energy_relative = Energy(1e-7, Kelvin);
    let basis_recipe = TramBasisRecipe {
        l_max: 10,
        n_max: 10,
        ..Default::default()
    };

    distribute!(
        || linspace(0., 2000., 64),
        |mag_field: f64| {
            let start = Instant::now();
            println!("magnetic field {mag_field:.1}");

            let mut atoms = get_particles(energy_relative, projection);
            let alkali_problem = get_problem(&atoms, &basis_recipe);

            let alkali_problem = alkali_problem.scattering_for(mag_field);
            let mut asymptotic = alkali_problem.asymptotic;
            asymptotic.entrance = entrance;
            atoms.insert(asymptotic);
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(5.0, Direction::Outwards, (1.001 * &id, 1.002 * &id));

            let eq = CoupledEquation::from_particles(potential, &atoms);
            let step_rule = LocalWavelengthStepRule::new(4e-3, f64::INFINITY, 400.);
            let mut numerov = MultiRNumerov::new(eq, boundary, step_rule);

            numerov.propagate_to(1500.);

            let scattering = numerov.s_matrix().get_scattering_length().re;

            let elapsed = start.elapsed();
            println!(
                "magnetic field {mag_field:.1}, done in: {:.2}",
                elapsed.as_secs_f64()
            );

            [mag_field, scattering]
        },
        |scatterings: Vec<[f64; 2]>| {
            let data = vec![
                scatterings.iter().map(|x| x[0]).collect(),
                scatterings.iter().map(|x| x[1]).collect(),
            ];

            save_data(
                &format!("srf_rb_n_max_{}_ground", basis_recipe.n_max),
                "mag_field\tscattering_length",
                &data,
            )
            .unwrap();
        }
    );
}

fn get_particles(energy: Energy<impl EnergyUnit>, projection: HalfI32) -> Particles {
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
        .with_hyperfine_coupling(Energy(6.83468261090429 / 2., GHz).to_au());

    let hifi_problem = DoubleHifiProblemBuilder::new(hifi_srf, hifi_rb).with_projection(projection);
    particles.insert(hifi_problem);

    particles
}

fn get_problem(
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

fn read_extended(max_degree: u32) -> [PotentialArray; 2] {
    let filename = "Rb_SrF/pot.data.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f = File::open(&path)
        .unwrap_or_else(|_| panic!("couldn't find potential in provided path {path:?}"));
    let f = BufReader::new(f);

    let filename = "Rb_SrF/casscf_ex.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    let f2 = File::open(&path)
        .unwrap_or_else(|_| panic!("couldn't find potential in provided path {path:?}"));
    let f2 = BufReader::new(f2);

    let angle_count = 1 + 180 / 5;
    let r_count = 30;
    let mut values_triplet = Mat::zeros(r_count, angle_count);
    let mut values_exch = Mat::zeros(r_count, angle_count);
    let mut distances = vec![0.; r_count];
    let angles: Vec<f64> = (0..=180).step_by(5).map(|x| x as f64 / 180. * PI).collect();

    for ((i, line_triplet), line_diff) in f.lines().skip(1).enumerate().zip(f2.lines().skip(1)) {
        let line_triplet = line_triplet.unwrap();
        let splitted_triplet: Vec<&str> = line_triplet.split_whitespace().collect();

        let r: f64 = splitted_triplet[0].parse().unwrap();
        let value: f64 = splitted_triplet[1].parse().unwrap();

        let angle_index = i / r_count;
        let r_index = i % r_count;

        if angle_index > 0 {
            assert!(distances[r_index] == Distance(r, Angstrom).to_au())
        }

        let line_diff = line_diff.unwrap();
        let splitted_diff: Vec<&str> = line_diff.split_whitespace().collect();

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
    let f = File::open(&path)
        .unwrap_or_else(|_| panic!("couldn't find potential in provided path {path:?}"));
    let mut f = BufReader::new(f);

    let mut weights = String::new();
    f.read_line(&mut weights).unwrap();
    let weights: Vec<f64> = weights
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
            .iter()
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
        potentials_triplet.push((lambda, values_triplet));

        let values_singlet = distances_extended
            .iter()
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
        potentials_singlet.push((lambda, values_singlet));
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

fn get_interpolated(
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
        .zip(potentials_far)
        .map(|((lambda, near), far)| {
            let combined = TransitionedPotential::new(near, far, transition);

            (lambda, combined)
        })
        .collect()
}
