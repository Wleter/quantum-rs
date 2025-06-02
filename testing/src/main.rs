use argmin::{core::{CostFunction, Error, Executor, State}, solver::{neldermead::NelderMead, particleswarm::ParticleSwarm, simulatedannealing::SimulatedAnnealing}};

struct MyProblem {}

impl CostFunction for MyProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok((p[0] - 4.0).abs().sqrt() + (p[1] - 10.).abs().sqrt())
    }
}

fn main() {
    let cost = MyProblem {};

    let solver = NelderMead::new(vec![vec![-50., -100.], vec![50., -100.], vec![0., 100.]])
        .with_sd_tolerance(1e-5).unwrap();

    let res = Executor::new(cost, solver)
        .configure(|state| 
            state.max_iters(100)
                .target_cost(0.)
    )
    .run()
    .unwrap();

    println!("{}", res);
    let best = res.state().get_best_param().unwrap();
    println!("{:?}", best);

}
