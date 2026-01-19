# rsoderh SIMD

I've implemented a 4x4 matrix multiplication library.

## Benchmarks

Each benchmark performs a single matrix multiplication operation a number of million times. The benchmarks measure the time for that single matrix multiplication. The benchmarks are performed with both a development and release build.

```shell
cargo bench --profile=dev
```

| Benchmark (dev) | Average time | Confidence interval |
| --- | --- | --- |
| SISD | 3.1918 µs | 3.1725 µs - 3.2175 µs |
| SIMD | 532.93 ns | 530.95 ns - 535.03 ns |

```shell
cargo bench --profile=release
```

| Benchmark (release) | Average time | Confidence interval |
| --- | --- | --- |
| SISD | 5.1316 ns | 5.0928 ns - 5.1758 µs |
| SIMD | 5.0920 ns | 5.0700 ns - 5.1158 ns |

So on development builds the SIMD implementation is significantly faster than the SISD one. This might not be a very fair comparison, as the SIMD implementation contain a lot of unsafe code, which skip safety checks which the SISD implementation performs.

These differences are reduced in the release build, with no real difference between the implementations' performance. When inspecting the assembly with compiler explorer the SISD implementation was optimized to use 128-bit vector instructions, while the SIMD implementation only uses 256-bit vector extensions.


