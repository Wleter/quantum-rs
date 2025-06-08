use std::io::{self, BufRead};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use rayon::prelude::*;

fn main() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let c = cancelled.clone();

    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(input) if input.trim() == "stop" => {
                    c.store(true, Ordering::SeqCst);
                    break;
                }
                _ => {}
            }
        }
    });

    let data: Vec<f64> = (0..300).map(|x| x as f64).collect();

    let results: Vec<f64> = data
        .par_iter()
        .filter_map(|&x| {
            if cancelled.load(Ordering::SeqCst) {
                None
            } else {
                // Simulate work
                println!("started {x}");
                thread::sleep(Duration::from_secs(1));
                Some(2. * x)
            }
        })
        .collect();

    println!("{}", results.len());
    println!("{}", data.len());
    println!("{}", data[0..results.len()].len())
}