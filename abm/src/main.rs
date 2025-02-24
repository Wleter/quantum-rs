use problems::Problems;
use quantum::problem_selector::{ProblemSelector, get_args};

pub mod consts;
pub mod problems;

fn main() {
    Problems::select(&mut get_args());
}
