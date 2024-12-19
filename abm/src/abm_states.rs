use clebsch_gordan::half_integer::{HalfI32, HalfU32};
use faer::Mat;
use quantum::{
    cast_variant,
    states::{
        operator::Operator,
        spins::{Spin, SpinOperators},
        StatesBasis,
    },
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStates {
    ElectronSpin(HalfU32),
    NuclearSpin(HalfU32),
}

impl HifiStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<HifiStates, HalfI32>,
        basis_sep: &StatesBasis<HifiStatesSep, HalfI32>,
    ) -> Operator<Mat<f64>> {
        let operator = hifi_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStatesSep {
    ElSpin1(HalfU32),
    ElSpin2(HalfU32),
    NuclearSpin1(HalfU32),
    NuclearSpin2(HalfU32),
}

impl HifiStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<HifiStatesSep, HalfI32>,
        basis: &StatesBasis<HifiStates, HalfI32>,
    ) -> Operator<Mat<f64>> {
        hifi_convert_to_comb(basis_sep, basis)
    }
}

fn hifi_convert_to_comb(
    basis_sep: &StatesBasis<HifiStatesSep, HalfI32>,
    basis: &StatesBasis<HifiStates, HalfI32>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep.variants[0], HifiStatesSep::ElSpin1);
        let s1 = Spin::new(s1, sep.values[0]);

        let s2 = cast_variant!(sep.variants[1], HifiStatesSep::ElSpin2);
        let s2 = Spin::new(s2, sep.values[1]);

        let i1 = cast_variant!(sep.variants[2], HifiStatesSep::NuclearSpin1);
        let i1 = Spin::new(i1, sep.values[2]);

        let i2 = cast_variant!(sep.variants[3], HifiStatesSep::NuclearSpin2);
        let i2 = Spin::new(i2, sep.values[3]);

        let s_tot = cast_variant!(comb.variants[0], HifiStates::ElectronSpin);
        let s_tot = Spin::new(s_tot, comb.values[0]);

        let i_tot = cast_variant!(comb.variants[1], HifiStates::NuclearSpin);
        let i_tot = Spin::new(i_tot, comb.values[1]);

        SpinOperators::clebsch_gordan(s1, s2, s_tot) * SpinOperators::clebsch_gordan(i1, i2, i_tot)
    })
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStates {
    ElectronSpin(HalfU32),
    NuclearSpin(HalfU32),
    Vibrational,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStatesValues {
    Proj(HalfI32),
    Vib(i32)
}

impl ABMStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<ABMStates, ABMStatesValues>,
        basis_sep: &StatesBasis<ABMStatesSep, ABMStatesValues>,
    ) -> Operator<Mat<f64>> {
        let operator = abm_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStatesSep {
    ElSpin1(HalfU32),
    ElSpin2(HalfU32),
    NuclearSpin1(HalfU32),
    NuclearSpin2(HalfU32),
    Vibrational,
}

impl ABMStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<ABMStatesSep, ABMStatesValues>,
        basis: &StatesBasis<ABMStates, ABMStatesValues>,
    ) -> Operator<Mat<f64>> {
        abm_convert_to_comb(basis_sep, basis)
    }
}

fn abm_convert_to_comb(
    basis_sep: &StatesBasis<ABMStatesSep, ABMStatesValues>,
    basis: &StatesBasis<ABMStates, ABMStatesValues>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep.variants[0], ABMStatesSep::ElSpin1);
        let ms1 = cast_variant!(sep.values[0], ABMStatesValues::Proj);

        let s1 = Spin::new(s1, ms1);

        let s2 = cast_variant!(sep.variants[1], ABMStatesSep::ElSpin2);
        let ms2 = cast_variant!(sep.values[1], ABMStatesValues::Proj);
        let s2 = Spin::new(s2, ms2);

        let i1 = cast_variant!(sep.variants[2], ABMStatesSep::NuclearSpin1);
        let mi1 = cast_variant!(sep.values[2], ABMStatesValues::Proj);
        let i1 = Spin::new(i1, mi1);

        let i2 = cast_variant!(sep.variants[3], ABMStatesSep::NuclearSpin2);
        let mi2 = cast_variant!(sep.values[3], ABMStatesValues::Proj);
        let i2 = Spin::new(i2, mi2);

        let s_tot = cast_variant!(comb.variants[0], ABMStates::ElectronSpin);
        let ms_tot = cast_variant!(comb.values[0], ABMStatesValues::Proj);
        let s_tot = Spin::new(s_tot, ms_tot);

        let i_tot = cast_variant!(comb.variants[1], ABMStates::NuclearSpin);
        let mi_tot = cast_variant!(comb.values[1], ABMStatesValues::Proj);
        let i_tot = Spin::new(i_tot, mi_tot);

        if sep.values[4] == comb.values[2] {
            SpinOperators::clebsch_gordan(s1, s2, s_tot)
                * SpinOperators::clebsch_gordan(i1, i2, i_tot)
        } else {
            0.0
        }
    })
}
