use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsoderh_simd::matrix_sisd::Matrix4x4Sisd;

// fn multiply(matrix)

fn criterion_benchmark(c: &mut Criterion) {
    let matrix_a = Matrix4x4Sisd::from_rows([
        [1., 2., 0., 1.],
        [0., 1., 3., 2.],
        [4., 0., 1., 0.],
        [2., 1., 0., 1.],
    ]);
    let matrix_b = Matrix4x4Sisd::from_rows([
        [2., 1., 3., 0.],
        [1., 0., 2., 1.],
        [0., 1., 1., 2.],
        [3., 0., 0., 1.],
    ]);

    c.bench_function("Matrix SIMD", |b| {
        b.iter(|| &black_box(matrix_a.clone()) * &black_box(matrix_b.clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
