use std::{collections::VecDeque, panic};

/// Gets the arguments from the command line and returns them as a VecDeque
pub fn get_args() -> VecDeque<String> {
    let mut args = std::env::args();
    // get rid of the first argument which is the program name
    args.next();

    args.collect()
}

/// Trait for selecting a problem to run
pub trait ProblemSelector {
    /// Name of the problem.
    const NAME: &'static str;

    /// Vector of all available problems to choose
    fn list() -> Vec<&'static str>;

    /// Given a problem number, run the problem from the list of problems.
    /// Problem can be [`select(args)`] function of other [ProblemSelector]
    fn methods(number: &str, args: &mut VecDeque<String>);

    /// Select a problem to run preselected or from user input
    /// The problem can be run with -1 to run all problems.
    fn select(args: &mut VecDeque<String>) {
        let arg = args.pop_front();

        match arg {
            Some(arg) => {
                if arg == "-1" {
                    select_many(&Self::list(), Self::methods);
                    return;
                }

                Self::methods(&arg.to_string(), args)
            }
            None => {
                println!();
                println!("Provide a problem number:");
                println!("-1: run all problems");

                let problems = Self::list();
                for (i, problem) in problems.iter().enumerate() {
                    println!("{}: {}", i, problem);
                }

                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "-1" {
                    select_many(&Self::list(), Self::methods);
                    return;
                }

                Self::methods(input, args)
            }
        }

        fn select_many(
            list: &[&'static str],
            methods: impl Fn(&str, &mut VecDeque<String>) + std::panic::RefUnwindSafe,
        ) {
            let args = VecDeque::from(vec!["-1".to_string()]);

            for i in 0..list.len() {
                let result = panic::catch_unwind(|| (methods)(&i.to_string(), &mut args.clone()));

                if result.is_err() {
                    println!("Problem {} failed", i);
                }
            }
        }
    }
}

#[macro_export]
macro_rules! problems_impl {
    ($selector:ty, $name:expr, $($problem_type:expr => $method:expr),* $(,)?) => {
        impl $crate::problem_selector::ProblemSelector for $selector {
            const NAME: &'static str = $name;

            fn list() -> Vec<&'static str> {
                vec![$($problem_type),*]
            }

            // todo! can be done better
            #[allow(unused_assignments)]
            fn methods(number: &str, args: &mut std::collections::VecDeque<String>) {
                let name_list = Self::list();

                let mut i: i32 = 0;
                $(
                    if &i.to_string() == number {
                        println!("Chose problem: {}", name_list[i as usize]);
                        $method(args);
                        return;
                    }

                    i += 1;
                )*

                panic!("Not found");
            }
        }
    };
}

#[cfg(test)]
mod test {
    use crate::problem_selector::{ProblemSelector, get_args};

    struct TestProblems;

    problems_impl!(TestProblems, "test",
        "test1" => |_| println!("test1"),
        "test2" => |_| println!("test2"),
        "test3" => |args| println!("{:?}", args)
    );

    #[test]
    fn problem_selector() {
        println!("{:?}", TestProblems::list());
        TestProblems::methods("2", &mut get_args());
    }
}
