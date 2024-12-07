use problems::Problems;
use quantum::problem_selector::{get_args, ProblemSelector};

pub mod consts;
pub mod problems;

fn main() {
    Problems::select(&mut get_args());
}
