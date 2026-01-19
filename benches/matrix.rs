use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsoderh_simd::{matrix_simd::Matrix4x4Simd, matrix_sisd::Matrix4x4Sisd};

// fn multiply(matrix)

fn criterion_benchmark(c: &mut Criterion) {
    let matrix_a_src = Matrix4x4Sisd::from_rows([
        [1., 2., 0., 1.],
        [0., 1., 3., 2.],
        [4., 0., 1., 0.],
        [2., 1., 0., 1.],
    ]);
    let matrix_b_src = Matrix4x4Sisd::from_rows([
        [2., 1., 3., 0.],
        [1., 0., 2., 1.],
        [0., 1., 1., 2.],
        [3., 0., 0., 1.],
    ]);

    let matrix_a = matrix_a_src.clone();
    let matrix_b = matrix_b_src.clone();

    c.bench_function("matrix_sisd", |b| {
        b.iter(|| black_box(&matrix_a) * black_box(&matrix_b))
    });

    let matrix_a = Matrix4x4Simd::from(matrix_a_src);
    let matrix_b = Matrix4x4Simd::from(matrix_b_src);

    c.bench_function("matrix_simd", |b| {
        b.iter(|| black_box(&matrix_a) * black_box(&matrix_b))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
