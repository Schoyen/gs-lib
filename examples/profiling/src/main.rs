use std::time::Instant;

use gs_lib::one_dim::{construct_overlap_matrix_elements, G1D};

fn profile_g1d_construction(n: u32) {
    println!("\n=====Profiling G1D construction=====");
    let now_full = Instant::now();

    for i in 0..n {
        let now_g = Instant::now();
        let _g1d = G1D::new(i, 1.0, 0.0, 'x');
        println!("Time G1D({}): {} ns", i, now_g.elapsed().as_nanos());
    }
    println!("Total time: {} ns", now_full.elapsed().as_nanos());
    println!("=====================================");
}

fn profile_g1d_overlap(n: u32) {
    println!("\n=====Profiling G1D overlap=====");
    let now = Instant::now();
    let gaussians = (0..n)
        .map(|i| G1D::new(i, 1.0, 0.0, 'x'))
        .collect::<Vec<G1D>>();
    println!(
        "Time constructing Gaussians: {} sec",
        (now.elapsed().as_nanos() as f64) / 1e9
    );

    let now = Instant::now();
    let _s = construct_overlap_matrix_elements(&gaussians);
    println!(
        "Time constructing overlap matrix: {} sec",
        (now.elapsed().as_nanos() as f64) / 1e9
    );

    println!("===============================");
}

fn main() {
    profile_g1d_construction(27);
    profile_g1d_overlap(16);
}
