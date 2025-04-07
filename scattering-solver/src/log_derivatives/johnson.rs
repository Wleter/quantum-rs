use faer::{MatMut, MatRef};

use super::{LogDerivative, LogDerivativeReference};

pub type JohnsonLogDerivative<'a, S> = LogDerivative<'a, S, Johnson>;

pub struct Johnson;

impl LogDerivativeReference for Johnson {
    fn w_ref(_w_c: MatRef<f64>, mut w_ref: MatMut<f64>) {
        w_ref.fill(0.);
    }

    fn imbedding1(h: f64, _w_ref: MatRef<f64>, mut out: MatMut<f64>) {
        out.fill(0.);

        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .for_each(|y1| *y1 = 1.0 / h);
    }

    fn imbedding2(h: f64, _w_ref: MatRef<f64>, mut out: MatMut<f64>) {
        out.fill(0.);

        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .for_each(|y2| *y2 = 1.0 / h);
    }

    #[inline]
    fn imbedding3(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: MatRef<f64>, out: MatMut<f64>) {
        Self::imbedding1(h, w_ref, out);
    }
}

#[cfg(test)]
mod test {
    use faer::Mat;
    use quantum::{
        assert_approx_eq,
        params::{particle_factory::create_atom, particles::Particles},
        units::*,
    };

    use crate::{
        boundary::{Asymptotic, Boundary, Direction},
        log_derivatives::johnson::JohnsonLogDerivative,
        numerovs::LocalWavelengthStepRule,
        potentials::{
            dispersion_potential::Dispersion,
            gaussian_coupling::GaussianCoupling,
            multi_coupling::MultiCoupling,
            multi_diag_potential::Diagonal,
            pair_potential::PairPotential,
            potential::{MatPotential, Potential},
            potential_factory::create_lj,
        },
        propagator::{CoupledEquation, Propagator},
        utility::AngMomentum,
    };

    fn potential() -> impl MatPotential {
        let potential_lj1 = create_lj(Energy(0.002, Au), Distance(9., Au));
        let mut potential_lj2 = create_lj(Energy(0.0021, Au), Distance(8.9, Au));
        potential_lj2.add_potential(Dispersion::new(Energy(1., Kelvin).to_au(), 0));

        let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);

        let potential = Diagonal::<Mat<f64>, _>::from_vec(vec![potential_lj1, potential_lj2]);
        let coupling = MultiCoupling::<Mat<f64>, _>::new_neighboring(vec![coupling]);

        PairPotential::new(potential, coupling)
    }

    fn particles() -> Particles {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.insert(Asymptotic {
            centrifugal: vec![AngMomentum(0); 2],
            entrance: 0,
            channel_energies: vec![0., Energy(0.0021, Kelvin).to_au()],
            channel_states: Mat::identity(2, 2),
        });

        particles
    }

    #[test]
    fn test_scattering_johnson_log_derivative() {
        let particles = particles();
        let potential = potential();

        let boundary = Boundary::new_multi_vanishing(6.5, Direction::Outwards, potential.size());
        let eq = CoupledEquation::from_particles(&potential, &particles);

        let mut log_deriv =
            JohnsonLogDerivative::new(eq, boundary, LocalWavelengthStepRule::default());

        log_deriv.propagate_to(1500.0);
        let s_matrix = log_deriv.s_matrix();

        // values at which the result was correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -36.998695, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -8.697948e-13, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 1.720206e4, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 1.0356329e-23, 1e-6);
    }
}
