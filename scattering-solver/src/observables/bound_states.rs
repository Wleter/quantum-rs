use quantum::units::{Unit, energy_units::Energy, Au};

use crate::{boundary::{Boundary, Direction}, collision_params::CollisionParams, defaults::{DynDefaults, SingleDefaults}, numerovs::{propagator::{Numerov, Sampling, SamplingStorage, StepConfig}, ratio_numerov::RatioNumerov}, potentials::potential::Potential, types::DFMatrix};

pub struct SingleBounds<'a, P: Potential<Space = f64>> {
    collision_params: &'a mut CollisionParams<P>,
    r_range: (f64, f64),
    step_config: StepConfig,
}

impl<P: Potential<Space = f64>> SingleBounds<'_, P> {
    pub fn new(collision_params: &mut CollisionParams<P>, r_range: (f64, f64)) -> SingleBounds<P> {
        SingleBounds {
            collision_params,
            r_range,
            step_config: StepConfig::Variable(1.0, 0.01, 5.0),
        }
    }

    pub fn set_step_config(mut self, step_config: StepConfig) -> Self {
        self.step_config = step_config;
        self
    }

    /// Return ascending bound energies found in the given energy range
    pub fn bound_spectrum<U: Unit>(&mut self, energy_range: (Energy<U>, Energy<U>), err: Energy<U>) -> Vec<Energy<Au>> {
        let err = err.to_au();

        let lowest_energy = potential_minimum(self.collision_params, self.r_range) + err;
        let energy_range = (energy_range.0.to_au().max(lowest_energy + err), energy_range.1.to_au());
        let mut energies = Vec::new();

        self.collision_params.particles.internals.insert_value("energy", energy_range.0);
        let (diff, mut nodes_min) = self.bound_diffs();
        if diff > 0.0 {
            nodes_min += 1;
        }

        self.collision_params.particles.internals.insert_value("energy", energy_range.1);
        let (diff, mut nodes_max) = self.bound_diffs();
        if diff < 0.0 {
            if nodes_max == 0 { return energies; }
            nodes_max -= 1;
        }

        for n in nodes_min..=nodes_max {
            energies.push(Energy(self.bin_search(energy_range, err, n as isize), Au));
        }

        energies
    }
    
    pub fn n_bound_energy<U: Unit>(&mut self, n_bound: isize, err: Energy<U>) -> Energy<Au> {
        let err = err.to_au();

        let upper_energy = self.collision_params.potential.value(&self.r_range.1);
        let lower_energy = potential_minimum(self.collision_params, self.r_range) + err;

        Energy(self.bin_search((lower_energy, upper_energy), err, n_bound), Au)
    }

    fn bin_search(&mut self, mut energy_range: (f64, f64), err: f64, n: isize) -> f64 {
        assert!(energy_range.1 > energy_range.0);

        self.collision_params.particles.internals.insert_value("energy", energy_range.1);
        let (diff, mut nodes_max) = self.bound_diffs(); 
        if diff < 0.0 {
            if nodes_max == 0 {
                panic!("Found 0 existing bound states");
            }
            nodes_max -= 1;
        }

        if (nodes_max as isize) < n || (nodes_max as isize) < -n - 1 {
            panic!("Found only {nodes_max} states");
        }
        
        let target_nodes = if n < 0 {
            nodes_max - n.unsigned_abs() + 1
        } else {
            n as usize
        };

        self.collision_params.particles.internals.insert_value("energy", energy_range.0);
        let (diff, mut nodes_min) = self.bound_diffs(); 
        if diff > 0.0 {
            nodes_min += 1;
        }

        if nodes_min > nodes_max {
            panic!("No bound states found with given energy range");
        }

        while energy_range.1 - energy_range.0 > err {
            let mid_energy = (energy_range.1 + energy_range.0) / 2.0;

            self.collision_params.particles.internals.insert_value("energy", mid_energy);
            let (diff, nodes) = self.bound_diffs();

            if nodes > target_nodes {
                energy_range.1 = mid_energy
            } else if nodes < target_nodes {
                energy_range.0 = mid_energy
            } else if diff > 0.0 {
                energy_range.1 = mid_energy;
            } else {
                energy_range.0 = mid_energy
            }
        }

        let mid_energy = (energy_range.1 + energy_range.0) / 2.0;

        self.collision_params.particles.internals.insert_value("energy", mid_energy);
        mid_energy
    }

    pub fn bound_wave(&self, sampling: Sampling) -> (Vec<f64>, Vec<f64>) {
        let mut numerov = RatioNumerov::new(self.collision_params)
            .set_step_config(self.step_config);
        let r_stop = numerov.propagation_distance(self.r_range);

        let inwards_boundary = Boundary::new(r_stop, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(self.r_range.0, Direction::Outwards, SingleDefaults::boundary());

        numerov.prepare(&inwards_boundary);

        let mut wave = 1e-50;
        let mut sampler = SamplingStorage::new(sampling, &numerov.r(), &wave, &r_stop);

        while *numerov.wave_last() > 1.0 && numerov.r() > self.r_range.0 {
            numerov.single_step();

            wave *= numerov.wave_last();
            sampler.sample(&numerov.r(), &wave);
        }

        let (mut rs_inwards, mut wave_inwards) = sampler.result();
        rs_inwards.reverse();
        wave_inwards.reverse();

        numerov.prepare(&outwards_boundary);
        let r_match = *rs_inwards.first().unwrap();
        let (mut rs, mut wave) = numerov.propagate_values(r_match, 1e-50, sampling);
        rs.pop();
        wave.pop();

        let normalization = wave_inwards.first().unwrap() / wave.last().unwrap();
        wave.iter_mut().for_each(|w| *w *= normalization);

        wave.extend(wave_inwards);
        rs.extend(rs_inwards);

        (rs, wave)
    }

    pub fn bound_diff_dependence<U: Unit>(&mut self, energies: &[Energy<U>]) -> (Vec<f64>, Vec<usize>) {
        let mut bound_differences = Vec::new();
        let mut node_counts = Vec::new();
        for energy in energies {
            self.collision_params.particles.internals.insert_value("energy", energy.to_au());
            
            let (bound_difference, node_count) = self.bound_diffs();
            bound_differences.push(bound_difference);
            node_counts.push(node_count);
        }
        
        (bound_differences, node_counts)
    }

    pub fn bound_diffs(&self) -> (f64, usize) {   
        let mut numerov = RatioNumerov::new(self.collision_params)
            .set_step_config(self.step_config);
        let r_stop = numerov.propagation_distance(self.r_range);

        let inwards_boundary = Boundary::new(r_stop, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(self.r_range.0, Direction::Outwards, SingleDefaults::boundary());

        numerov.prepare(&inwards_boundary);

        while *numerov.wave_last() > 1.0 && numerov.r() > self.r_range.0 {
            numerov.single_step();
        }
        let inwards_result = numerov.result();
        
        numerov.prepare(&outwards_boundary);
        let r_match = inwards_result.r_last;
        
        let node_count = numerov.propagate_node_counting(r_match);
        numerov.propagate_to(r_match);
        let outwards_result = numerov.result();

        ((1.0 / outwards_result.wave_ratio - 1.0) * (1.0 / outwards_result.dr.abs()) 
            - (inwards_result.wave_ratio - 1.0) * (1.0 / inwards_result.dr.abs()), node_count)
    }
}

fn potential_minimum<P: Potential<Space = f64>>(collision_params: &CollisionParams<P>, r_range: (f64, f64)) -> f64 {
    let mut numerov = RatioNumerov::new(collision_params);
    numerov.potential_minimum(r_range)
}

fn generalized_minimum<P: Potential<Space = DFMatrix>>(collision_params: &CollisionParams<P>, r_range: (f64, f64)) -> f64 {
    let mut numerov = RatioNumerov::new_dyn(collision_params);
    numerov.potential_minimum(r_range)
}

pub struct MultiBounds<'a, P: Potential<Space = DFMatrix>> {
    collision_params: &'a mut CollisionParams<P>,
    r_range: (f64, f64),
    step_config: StepConfig,
}

impl<P: Potential<Space = DFMatrix>> MultiBounds<'_, P> {
    pub fn new(collision_params: &mut CollisionParams<P>, r_range: (f64, f64)) -> MultiBounds<P> {
        MultiBounds {
            collision_params,
            r_range,
            step_config: StepConfig::Variable(1.0, 0.01, 5.0),
        }
    }

    pub fn set_step_config(mut self, step_config: StepConfig) -> Self {
        self.step_config = step_config;
        self
    }

    /// Return ascending bound energies found in the given energy range
    pub fn bound_spectrum<U: Unit>(&mut self, energy_range: (Energy<U>, Energy<U>), err: Energy<U>) -> Vec<Energy<Au>> {
        let err = err.to_au();

        let lowest_energy = generalized_minimum(self.collision_params, self.r_range) + err;
        let energy_range = (energy_range.0.to_au().min(lowest_energy) + err, energy_range.1.to_au());
        let mut energies = Vec::new();

        self.collision_params.particles.internals.insert_value("energy", energy_range.0);
        let (diff, mut nodes_min) = self.bound_diffs();
        if diff > 0.0 {
            nodes_min += 1;
        }

        self.collision_params.particles.internals.insert_value("energy", energy_range.1);
        let (diff, mut nodes_max) = self.bound_diffs();
        if diff < 0.0 {
            if nodes_max == 0 { return energies; }
            nodes_max -= 1;
        }

        for n in nodes_min..=nodes_max {
            energies.push(Energy(self.bin_search(energy_range, err, n as isize), Au));
        }

        energies
    }
    
    pub fn n_bound_energy<U: Unit>(&mut self, n_bound: isize, err: Energy<U>) -> Energy<Au> {
        let err = err.to_au();

        let upper_energy = self.collision_params.potential.value(&self.r_range.1).max();
        let lower_energy = generalized_minimum(self.collision_params, self.r_range) + err;

        Energy(self.bin_search((lower_energy, upper_energy), err, n_bound), Au)
    }

    // unsafe because we assume there exist a bound state
    fn bin_search(&mut self, mut energy_range: (f64, f64), err: f64, n: isize) -> f64 {
        assert!(energy_range.1 > energy_range.0);

        self.collision_params.particles.internals.insert_value("energy", energy_range.1);
        let (diff, mut nodes_max) = self.bound_diffs(); 
        if diff < 0.0 {
            if nodes_max == 0 {
                panic!("Found 0 existing bound states");
            }
            nodes_max -= 1;
        }

        if (nodes_max as isize) < n || (nodes_max as isize) < -n - 1 {
            panic!("Found only {nodes_max} states");
        }
        
        let target_nodes = if n < 0 {
            nodes_max - n.unsigned_abs() + 1
        } else {
            n as usize
        };

        self.collision_params.particles.internals.insert_value("energy", energy_range.0);
        let (diff, mut nodes_min) = self.bound_diffs(); 
        if diff > 0.0 {
            nodes_min += 1;
        }

        if nodes_min > nodes_max {
            panic!("No bound states found with given energy range");
        }

        while energy_range.1 - energy_range.0 > err {
            let mid_energy = (energy_range.1 + energy_range.0) / 2.0;

            self.collision_params.particles.internals.insert_value("energy", mid_energy);
            let (diff, nodes) = self.bound_diffs();

            if nodes > target_nodes {
                energy_range.1 = mid_energy
            } else if nodes < target_nodes {
                energy_range.0 = mid_energy
            } else if diff > 0.0 {
                energy_range.1 = mid_energy;
            } else {
                energy_range.0 = mid_energy
            }
        }

        let mid_energy = (energy_range.1 + energy_range.0) / 2.0;

        self.collision_params.particles.internals.insert_value("energy", mid_energy);
        mid_energy
    }

    // pub fn bound_wave(&self, sampling: Sampling) -> (Vec<f64>, Vec<f64>) {
    //     let mut numerov = RatioNumerov::new_dyn(self.collision_params)
    //             .set_step_config(StepConfig::Variable(1.0, Some(5.0)));
    //     let r_stop = numerov.propagation_distance(self.r_range);

    //     let inwards_boundary = Boundary::new(r_stop, Direction::Inwards, SingleDefaults::boundary());
    //     let outwards_boundary = Boundary::new(self.r_range.0, Direction::Outwards, SingleDefaults::boundary());

    //     numerov.prepare(&inwards_boundary);

    //     let mut wave = 1e-50;
    //     let mut sampler = SamplingStorage::new(sampling, &numerov.r(), &wave, &r_stop);

    //     while *numerov.wave_last() > 1.0 && numerov.r() > self.r_range.0 {
    //         numerov.single_step();

    //         wave *= numerov.wave_last();
    //         sampler.sample(&numerov.r(), &wave);
    //     }

    //     let (mut rs_inwards, mut wave_inwards) = sampler.result();
    //     rs_inwards.reverse();
    //     wave_inwards.reverse();

    //     numerov.prepare(&outwards_boundary);
    //     let r_match = *rs_inwards.first().unwrap();
    //     let (mut rs, mut wave) = numerov.propagate_values(r_match, 1e-50, sampling);
    //     rs.pop();
    //     wave.pop();

    //     let normalization = wave_inwards.first().unwrap() / wave.last().unwrap();
    //     wave.iter_mut().for_each(|w| *w *= normalization);

    //     wave.extend(wave_inwards);
    //     rs.extend(rs_inwards);

    //     (rs, wave)
    // }

    pub fn bound_diff_dependence<U: Unit>(&mut self, energies: &[Energy<U>]) -> (Vec<f64>, Vec<usize>) {
        let mut bound_differences = Vec::new();
        let mut node_counts = Vec::new();
        for energy in energies {
            self.collision_params.particles.internals.insert_value("energy", energy.to_au());
            
            let (bound_difference, node_count) = self.bound_diffs();
            bound_differences.push(bound_difference);
            node_counts.push(node_count);
        }
        
        (bound_differences, node_counts)
    }

    pub fn bound_diffs(&self) -> (f64, usize) {   
        let mut numerov = RatioNumerov::new_dyn(self.collision_params)
            .set_step_config(self.step_config);
        let size = self.collision_params.potential.size();

        let inwards_boundary = Boundary::new(self.r_range.1, Direction::Inwards, DynDefaults::boundary(size));
        let outwards_boundary = Boundary::new(self.r_range.0, Direction::Outwards, DynDefaults::boundary(size));

        numerov.prepare(&inwards_boundary);

        while numerov.wave_last().determinant() > 1.0 && numerov.r() > self.r_range.0 {
            numerov.single_step();
        }
        let inwards_result = numerov.result();
        
        numerov.prepare(&outwards_boundary);
        let r_match = inwards_result.r_last;
        
        let node_count = numerov.propagate_node_counting(r_match);
        numerov.propagate_to(r_match);
        let outwards_result = numerov.result();

        ((outwards_result.wave_ratio - inwards_result.wave_ratio.try_inverse().unwrap()).determinant(), node_count)
    }
}