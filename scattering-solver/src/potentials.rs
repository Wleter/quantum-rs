pub mod composite_potential;
pub mod dispersion_potential;
pub mod function_potential;
pub mod gaussian_coupling;
pub mod morse_long_range;
pub mod pair_potential;
pub mod potential;
pub mod potential_factory;

pub mod masked_potential;
pub mod multi_coupling;
pub mod multi_diag_potential;

#[cfg(test)]
mod test {
    use quantum::{
        assert_approx_eq,
        units::{
            Au,
            energy_units::{CmInv, Energy},
        },
    };

    use crate::potentials::{
        dispersion_potential::Dispersion, function_potential::FunctionPotential,
        morse_long_range::MorseLongRangeBuilder, pair_potential::PairPotential,
        potential::SimplePotential,
    };

    #[test]
    fn test_potentials() {
        let const_potential = Dispersion::new(2., 0);
        assert_eq!(const_potential.value(1.), 2.);
        assert_eq!(const_potential.value(5.), 2.);

        let func_potential = FunctionPotential::new(|r, val| *val = r);
        assert_eq!(func_potential.value(1.), 1.);
        assert_eq!(func_potential.value(5.), 5.);
    }

    #[test]
    fn test_morse() {
        let d0 = Energy(0.002, CmInv);

        let tail = vec![
            Dispersion::new(1394.180, -6),
            Dispersion::new(83461.675549, -8),
            Dispersion::new(7374640.77, -10),
        ];

        let morse = MorseLongRangeBuilder::new(d0.to(Au), 7.880185, tail)
            .set_params(5, 3, 15.11784, 0.54)
            .set_betas(vec![-0.516129, -0.0980, 0.1133, -0.0251])
            .build();

        let r = 7.880185;
        let val = Energy(morse.value(r), Au).to(CmInv);

        assert!(
            d0.value() + val.value() < 1e-3,
            "Expected: {}, Got: {}",
            d0.value(),
            val.value()
        );
    }

    #[test]
    fn test_faer() {
        use faer::{Mat, mat};

        use crate::potentials::{multi_coupling::MultiCoupling, potential::Potential};

        use super::multi_diag_potential::Diagonal;

        const N: usize = 4;
        let potentials = (0..N).map(|x| Dispersion::new(x as f64, 0)).collect();

        let diagonal = Diagonal::<Mat<f64>, _>::from_vec(potentials);

        let expected = mat![
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 2., 0.],
            [0., 0., 0., 3.]
        ];

        let mut value = Mat::zeros(N, N);
        diagonal.value_inplace(1., &mut value);
        assert_approx_eq!(mat => value, expected, 1e-5);

        let potentials = (0..(N - 1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<Mat<f64>, _>::new(N, potentials);

        let expected = mat![
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 2.],
            [0., 0., 2., 0.]
        ];

        let mut value = Mat::zeros(4, 4);
        coupling.value_inplace(1., &mut value);
        assert_approx_eq!(mat => value, expected, 1e-5);

        let combined = PairPotential::new(diagonal, coupling);

        let expected = mat![
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 2., 2.],
            [0., 0., 2., 3.]
        ];

        let mut value = Mat::zeros(4, 4);
        combined.value_inplace(1., &mut value);
        assert_approx_eq!(mat => value, expected, 1e-5);
    }
}
