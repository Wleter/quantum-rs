use faer::Mat;
use quantum::{
    cast_variant,
    states::{
        StatesBasis,
        operator::Operator,
        spins::{Spin, SpinOperators},
    },
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStates {
    ElectronSpin(Spin),
    NuclearSpin(Spin),
}

impl HifiStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<HifiStates>,
        basis_sep: &StatesBasis<HifiStatesSep>,
    ) -> Operator<Mat<f64>> {
        let operator = hifi_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HifiStatesSep {
    ElSpin1(Spin),
    ElSpin2(Spin),
    NuclearSpin1(Spin),
    NuclearSpin2(Spin),
}

impl HifiStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<HifiStatesSep>,
        basis: &StatesBasis<HifiStates>,
    ) -> Operator<Mat<f64>> {
        hifi_convert_to_comb(basis_sep, basis)
    }
}

fn hifi_convert_to_comb(
    basis_sep: &StatesBasis<HifiStatesSep>,
    basis: &StatesBasis<HifiStates>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep[0], HifiStatesSep::ElSpin1);
        let s2 = cast_variant!(sep[1], HifiStatesSep::ElSpin2);
        let i1 = cast_variant!(sep[2], HifiStatesSep::NuclearSpin1);
        let i2 = cast_variant!(sep[3], HifiStatesSep::NuclearSpin2);

        let s_tot = cast_variant!(comb[0], HifiStates::ElectronSpin);
        let i_tot = cast_variant!(comb[1], HifiStates::NuclearSpin);

        SpinOperators::clebsch_gordan(s1, s2, s_tot) * SpinOperators::clebsch_gordan(i1, i2, i_tot)
    })
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStates {
    ElectronSpin(Spin),
    NuclearSpin(Spin),
    Vibrational(i32),
}

impl ABMStates {
    pub fn convert_to_separated(
        basis: &StatesBasis<ABMStates>,
        basis_sep: &StatesBasis<ABMStatesSep>,
    ) -> Operator<Mat<f64>> {
        let operator = abm_convert_to_comb(basis_sep, basis);

        Operator::new(operator.transpose().to_owned())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ABMStatesSep {
    ElSpin1(Spin),
    ElSpin2(Spin),
    NuclearSpin1(Spin),
    NuclearSpin2(Spin),
    Vibrational(i32),
}

impl ABMStatesSep {
    pub fn convert_to_combined(
        basis_sep: &StatesBasis<ABMStatesSep>,
        basis: &StatesBasis<ABMStates>,
    ) -> Operator<Mat<f64>> {
        abm_convert_to_comb(basis_sep, basis)
    }
}

fn abm_convert_to_comb(
    basis_sep: &StatesBasis<ABMStatesSep>,
    basis: &StatesBasis<ABMStates>,
) -> Operator<Mat<f64>> {
    Operator::get_transformation(basis_sep, basis, |sep, comb| {
        let s1 = cast_variant!(sep[0], ABMStatesSep::ElSpin1);
        let s2 = cast_variant!(sep[1], ABMStatesSep::ElSpin2);
        let i1 = cast_variant!(sep[2], ABMStatesSep::NuclearSpin1);
        let i2 = cast_variant!(sep[3], ABMStatesSep::NuclearSpin2);
        let vib_sep = cast_variant!(sep[4], ABMStatesSep::Vibrational);

        let s_tot = cast_variant!(comb[0], ABMStates::ElectronSpin);
        let i_tot = cast_variant!(comb[1], ABMStates::NuclearSpin);
        let vib_comb = cast_variant!(comb[2], ABMStates::Vibrational);

        if vib_sep == vib_comb {
            SpinOperators::clebsch_gordan(s1, s2, s_tot)
                * SpinOperators::clebsch_gordan(i1, i2, i_tot)
        } else {
            0.0
        }
    })
}
