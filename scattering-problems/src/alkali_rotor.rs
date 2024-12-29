use abm::{get_hifi, get_zeeman_prop, utility::diagonalize, HifiProblemBuilder};
use clebsch_gordan::{half_i32, half_integer::{HalfI32, HalfU32}, half_u32};
use faer::Mat;
use quantum::{cast_variant, params::{particle::Particle, particle_factory::RotConst}, states::{operator::Operator, spins::spin_projections, state::State, state_type::StateType, States, StatesBasis}};

use crate::{get_aniso_hifi, get_spin_rot, utility::{AnisoHifi, GammaSpinRot, RotorJTotMax, RotorJMax, RotorLMax}};

#[derive(Clone)]
pub struct AlkaliRotorProblemBuilder {
    hifi_problem: HifiProblemBuilder,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinRotor {
    /// (l, j, j_tot)
    Angular((u32, u32, u32)),
    RotorS(HalfU32),
    RotorI(HalfU32),
}

impl AlkaliRotorProblemBuilder {
    pub fn new(hifi_problem: HifiProblemBuilder) -> Self {
        assert!(hifi_problem.s == half_u32!(1/2));

        Self {
            hifi_problem,
        }
    }

    pub fn build(self, particle: &Particle) -> AlkaliRotorProblem {
        let l_max = particle.get::<RotorLMax>().expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particle.get::<RotorJMax>().expect("Did not find RotorJMax parameter in particles").0;
        let j_tot_max = particle.get::<RotorJTotMax>().map_or(0, |x| x.0);
        // todo! change to rotor particle having RotConst
        let rot_const = particle.get::<RotConst>().expect("Did not find RotConst parameter in the particles").0;
        let gamma_spin_rot = particle.get::<GammaSpinRot>().unwrap_or(&GammaSpinRot(0.)).0;
        let aniso_hifi = particle.get::<AnisoHifi>().unwrap_or(&AnisoHifi(0.)).0;
        
        let ls: Vec<u32> = (0..=l_max).collect();
        
        let mut angular_states = vec![];
        for j_tot in 0..=j_tot_max {
            for &l in &ls {
                let j_lower = (l as i32 - j_tot as i32).unsigned_abs().min(j_max);
                let j_upper = (l + j_tot).min(j_max);
                
                for j in j_lower..=j_upper {
                    let projections = spin_projections(HalfU32::from_doubled(2 * j_tot));
                    let state = State::new(SpinRotor::Angular((l, j, j_tot)), projections);
                    angular_states.push(state);
                }
            }
        }

        let s_rotor = State::new(
            SpinRotor::RotorS(self.hifi_problem.s),
            spin_projections(self.hifi_problem.s),
        );
        let i_rotor = State::new(
            SpinRotor::RotorI(self.hifi_problem.i),
            spin_projections(self.hifi_problem.i),
        );

        let mut states = States::default();
        states.push_state(StateType::Sum(angular_states))
            .push_state(StateType::Irreducible(s_rotor))
            .push_state(StateType::Irreducible(i_rotor));

        let basis = match self.hifi_problem.total_projection {
            Some(m_tot) => {
                states.iter_elements()
                    .filter(|els| {
                        els.values.iter().copied().sum::<HalfI32>() == m_tot
                    })
                    .collect()
            },
            None => states.get_basis(),
        };

        let j_centrifugal = Operator::from_diagonal_mel(&basis, [SpinRotor::Angular((0, 0, 0))], |[ang]| {
            let (_, j, _) = cast_variant!(ang.0, SpinRotor::Angular);

            rot_const * (j * (j + 1)) as f64
        });

        let mut hifi = Mat::zeros(basis.len(), basis.len());
        if let Some(a_hifi) = self.hifi_problem.a_hifi {
            hifi += get_hifi!(basis, SpinRotor::RotorS, SpinRotor::RotorI, a_hifi).as_ref()
        }

        let mut zeeman_prop = get_zeeman_prop!(basis, SpinRotor::RotorS, self.hifi_problem.gamma_e).into_backed();
        if let Some(gamma_i) = self.hifi_problem.gamma_i {
            zeeman_prop += get_zeeman_prop!(basis, SpinRotor::RotorI, gamma_i).as_ref()
        }

        let spin_rot = get_spin_rot!(&basis, SpinRotor::Angular, SpinRotor::RotorS, gamma_spin_rot);

        let aniso_hifi = get_aniso_hifi!(&basis, SpinRotor::Angular, 
            SpinRotor::RotorS, SpinRotor::RotorI, aniso_hifi);

        AlkaliRotorProblem {
            basis,
            mag_inv: hifi + aniso_hifi.as_ref() + j_centrifugal.as_ref() + spin_rot.as_ref(),
            mag_prop: zeeman_prop,
        }
    }
}


pub struct AlkaliRotorProblem {
    pub basis: StatesBasis<SpinRotor, HalfI32>,
    mag_inv: Mat<f64>,
    mag_prop: Mat<f64>
}

impl AlkaliRotorProblem {
    pub fn levels_at_field(&self, mag_field: f64) -> (Vec<f64>, Mat<f64>) {
        let internal = &self.mag_inv + mag_field * &self.mag_prop;

        diagonalize(internal.as_ref())
    }
}
