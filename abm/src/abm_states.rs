use faer::Mat;
use quantum::{
    cast_variant,
    states::{
        operator::Operator,
        spins::{DoubleSpin, SpinOperators},
        StatesBasis,
    },
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStates {
    ElectronDSpin(u32),
    NuclearDSpin(u32),
}

impl HifiStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<HifiStates, i32>,
        basis_sep: &StatesBasis<HifiStatesSep, i32>,
    ) -> Operator<Mat<f64>> {
        let operator = hifi_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStatesSep {
    ElDSpin1(u32),
    ElDSpin2(u32),
    NuclearDSpin1(u32),
    NuclearDSpin2(u32),
}

impl HifiStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<HifiStatesSep, i32>,
        basis: &StatesBasis<HifiStates, i32>,
    ) -> Operator<Mat<f64>> {
        hifi_convert_to_comb(basis_sep, basis)
    }
}

fn hifi_convert_to_comb(
    basis_sep: &StatesBasis<HifiStatesSep, i32>,
    basis: &StatesBasis<HifiStates, i32>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep.variants[0], HifiStatesSep::ElDSpin1);
        let s1 = DoubleSpin(s1, sep.values[0]);

        let s2 = cast_variant!(sep.variants[1], HifiStatesSep::ElDSpin2);
        let s2 = DoubleSpin(s2, sep.values[1]);

        let i1 = cast_variant!(sep.variants[2], HifiStatesSep::NuclearDSpin1);
        let i1 = DoubleSpin(i1, sep.values[2]);

        let i2 = cast_variant!(sep.variants[3], HifiStatesSep::NuclearDSpin2);
        let i2 = DoubleSpin(i2, sep.values[3]);

        let s_tot = cast_variant!(comb.variants[0], HifiStates::ElectronDSpin);
        let s_tot = DoubleSpin(s_tot, comb.values[0]);

        let i_tot = cast_variant!(comb.variants[1], HifiStates::NuclearDSpin);
        let i_tot = DoubleSpin(i_tot, comb.values[1]);

        SpinOperators::clebsch_gordan(s1, s2, s_tot) * SpinOperators::clebsch_gordan(i1, i2, i_tot)
    })
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStates {
    ElectronDSpin(u32),
    NuclearDSpin(u32),
    Vibrational,
}

impl ABMStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<ABMStates, i32>,
        basis_sep: &StatesBasis<ABMStatesSep, i32>,
    ) -> Operator<Mat<f64>> {
        let operator = abm_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStatesSep {
    ElDSpin1(u32),
    ElDSpin2(u32),
    NuclearDSpin1(u32),
    NuclearDSpin2(u32),
    Vibrational,
}

impl ABMStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<ABMStatesSep, i32>,
        basis: &StatesBasis<ABMStates, i32>,
    ) -> Operator<Mat<f64>> {
        abm_convert_to_comb(basis_sep, basis)
    }
}

fn abm_convert_to_comb(
    basis_sep: &StatesBasis<ABMStatesSep, i32>,
    basis: &StatesBasis<ABMStates, i32>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep.variants[0], ABMStatesSep::ElDSpin1);
        let s1 = DoubleSpin(s1, sep.values[0]);

        let s2 = cast_variant!(sep.variants[1], ABMStatesSep::ElDSpin2);
        let s2 = DoubleSpin(s2, sep.values[1]);

        let i1 = cast_variant!(sep.variants[2], ABMStatesSep::NuclearDSpin1);
        let i1 = DoubleSpin(i1, sep.values[2]);

        let i2 = cast_variant!(sep.variants[3], ABMStatesSep::NuclearDSpin2);
        let i2 = DoubleSpin(i2, sep.values[3]);

        let s_tot = cast_variant!(comb.variants[0], ABMStates::ElectronDSpin);
        let s_tot = DoubleSpin(s_tot, comb.values[0]);

        let i_tot = cast_variant!(comb.variants[1], ABMStates::NuclearDSpin);
        let i_tot = DoubleSpin(i_tot, comb.values[1]);

        if sep.values[4] == comb.values[2] {
            SpinOperators::clebsch_gordan(s1, s2, s_tot)
                * SpinOperators::clebsch_gordan(i1, i2, i_tot)
        } else {
            0.0
        }
    })
}
