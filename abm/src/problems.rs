use quantum::problems_impl;
use simple_abm::SimpleABM;

use self::{
    hifi_double::HifiDouble, hifi_single::HifiSingle, lithium_potassium::LithiumPotassium,
    potassium_bound::PotassiumBound,
};

mod hifi_double;
mod hifi_single;
mod lithium_potassium;
mod potassium_bound;
mod simple_abm;

pub struct Problems;

problems_impl!(Problems, "ABM tests",
    "atom hyperfine" => |_| HifiSingle::run(),
    "two atom hyperfine" => |_| HifiDouble::run(),
    "simple abm" => |_| SimpleABM::run(),
    "ABM potassium" => PotassiumBound::select,
    "ABM lithium-potassium" => LithiumPotassium::select
);
