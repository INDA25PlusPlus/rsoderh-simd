use std::{arch::x86_64, fmt::Debug, ops};

use approx::AbsDiffEq;

use crate::matrix_sisd::Matrix4x4Sisd;

#[derive(Copy, Clone, PartialEq, bytemuck::AnyBitPattern)]
#[repr(C, align(32))]
pub struct Matrix4x4Simd([[f32; 4]; 4]);

impl Matrix4x4Simd {
    pub const ZERO: Self = Self([[0.; 4]; 4]);

    pub const IDENTITY: Self = Self([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]);

    pub fn new(rows: [[f32; 4]; 4]) -> Self {
        Self(rows)
    }

    pub fn from_rows(rows: impl IntoIterator<Item = impl IntoIterator<Item = f32>>) -> Self {
        Self(
            rows.into_iter()
                .map(|iter| {
                    iter.into_iter()
                        .collect::<Box<[_]>>()
                        .as_ref()
                        .try_into()
                        .unwrap()
                })
                .collect::<Box<[_]>>()
                .as_ref()
                .try_into()
                .unwrap(),
        )
    }

    pub fn rows(&self) -> &[[f32; 4]; 4] {
        &self.0
    }

    /// Returns the rows of this matrix packed into two 256 bit vector registers, with the first
    /// containing rows 0 and 1, and the second containing rows 2 and 3.
    pub fn rows_m256(&self) -> (x86_64::__m256, x86_64::__m256) {
        // SAFETY: index 8 is less than `self.flat_cells().len()` = 16
        let (rows_0_1, rows_2_3) = unsafe { self.flat_cells().split_at_unchecked(8) };

        // SAFETY: rows_x_x points to an array of length 8, with the same size and alignment as
        // `__m256`.
        let rows_0_1_m256: x86_64::__m256 =
            unsafe { *(rows_0_1.as_ptr() as *const x86_64::__m256) };
        let rows_2_3_m256: x86_64::__m256 =
            unsafe { *(rows_2_3.as_ptr() as *const x86_64::__m256) };

        (rows_0_1_m256, rows_2_3_m256)
    }

    /// Returns the matrix's cells as a slice in row-major order.
    pub fn flat_cells(&self) -> &[f32; 4 * 4] {
        match bytemuck::try_cast_ref(&self.0) {
            Ok(cells) => cells,
            // `self.0` has the same size and alignment as `[f32; 4 * 4]`.
            Err(_) => unreachable!(),
        }
    }

    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        Self(self.0.map(|row| row.map(&f)))
    }

    // Needs to be separate method since `target_feature` isn't supported in trait methods.
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "fma")]
    pub fn multiply(&self, rhs: &Self) -> Self {
        let self_rows = self.rows();
        let (rhs_rows_0_1, rhs_rows_2_3) = rhs.rows_m256();

        // Example matrices
        // self      rhs
        // 1 2 3 4 | 1 5 9 4
        // 5 6 7 8 | 2 6 1 5
        // 9 1 2 3 | 3 7 2 6
        // 4 5 6 7 | 4 8 3 7
        //
        // We fill
        // `self_column_0_rows_0_1` with [1 1 1 1 5 5 5 5],
        // `self_column_1_rows_0_1` with [2 2 2 2 6 6 6 6],
        // `self_column_2_rows_0_1` with [3 3 3 3 7 7 7 7],
        // `self_column_3_rows_0_1` with [4 4 4 4 8 8 8 8],
        // and
        // `rhs_rows_0_0` with [1 5 9 4 1 5 9 4],
        // `rhs_rows_1_1` with [2 6 1 5 2 6 1 5],
        // and so on.
        //
        // Then we multiply these with their corresponding rows of `rhs` and add the results
        // together. This gives us the values of the first two rows of the result matrix.

        let rhs_rows_0_0 =
            x86_64::_mm256_permute2f128_ps::<0b0000_0000>(rhs_rows_0_1, rhs_rows_0_1);
        let rhs_rows_1_1 =
            x86_64::_mm256_permute2f128_ps::<0b0001_0001>(rhs_rows_0_1, rhs_rows_0_1);
        let rhs_rows_2_2 =
            x86_64::_mm256_permute2f128_ps::<0b0000_0000>(rhs_rows_2_3, rhs_rows_2_3);
        let rhs_rows_3_3 =
            x86_64::_mm256_permute2f128_ps::<0b0001_0001>(rhs_rows_2_3, rhs_rows_2_3);

        let result_rows_0_1 = {
            let self_column_0_rows_0_1 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[0][0]),
                x86_64::_mm256_set1_ps(self_rows[1][0]),
            );
            let self_column_1_rows_0_1 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[0][1]),
                x86_64::_mm256_set1_ps(self_rows[1][1]),
            );
            let self_column_2_rows_0_1 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[0][2]),
                x86_64::_mm256_set1_ps(self_rows[1][2]),
            );
            let self_column_3_rows_0_1 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[0][3]),
                x86_64::_mm256_set1_ps(self_rows[1][3]),
            );

            let mut result_rows_0_1 = x86_64::_mm256_mul_ps(self_column_0_rows_0_1, rhs_rows_0_0);
            result_rows_0_1 =
                x86_64::_mm256_fmadd_ps(self_column_1_rows_0_1, rhs_rows_1_1, result_rows_0_1);
            result_rows_0_1 =
                x86_64::_mm256_fmadd_ps(self_column_2_rows_0_1, rhs_rows_2_2, result_rows_0_1);
            result_rows_0_1 =
                x86_64::_mm256_fmadd_ps(self_column_3_rows_0_1, rhs_rows_3_3, result_rows_0_1);
            result_rows_0_1
        };

        // We do the same but for rows 2 and 3
        let result_rows_2_3 = {
            let self_column_0_rows_2_3 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[2][0]),
                x86_64::_mm256_set1_ps(self_rows[3][0]),
            );
            let self_column_1_rows_2_3 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[2][1]),
                x86_64::_mm256_set1_ps(self_rows[3][1]),
            );
            let self_column_2_rows_2_3 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[2][2]),
                x86_64::_mm256_set1_ps(self_rows[3][2]),
            );
            let self_column_3_rows_2_3 = x86_64::_mm256_blend_ps::<0b1111_0000>(
                x86_64::_mm256_set1_ps(self_rows[2][3]),
                x86_64::_mm256_set1_ps(self_rows[3][3]),
            );

            let mut result_rows_2_3 = x86_64::_mm256_mul_ps(self_column_0_rows_2_3, rhs_rows_0_0);
            result_rows_2_3 =
                x86_64::_mm256_fmadd_ps(self_column_1_rows_2_3, rhs_rows_1_1, result_rows_2_3);
            result_rows_2_3 =
                x86_64::_mm256_fmadd_ps(self_column_2_rows_2_3, rhs_rows_2_2, result_rows_2_3);
            result_rows_2_3 =
                x86_64::_mm256_fmadd_ps(self_column_3_rows_2_3, rhs_rows_3_3, result_rows_2_3);
            result_rows_2_3
        };

        match bytemuck::try_cast([result_rows_0_1, result_rows_2_3]) {
            Ok(result) => result,
            // `[__m256; 2]` has the same size and alignment as `Matrix4x4Simd`.
            Err(_) => unreachable!(),
        }
    }
}

impl From<Matrix4x4Sisd> for Matrix4x4Simd {
    fn from(value: Matrix4x4Sisd) -> Self {
        Self(value.0)
    }
}

impl From<Matrix4x4Simd> for Matrix4x4Sisd {
    fn from(value: Matrix4x4Simd) -> Self {
        Self(value.0)
    }
}

impl ops::Index<(usize, usize)> for Matrix4x4Simd {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.1][index.0]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix4x4Simd {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.1][index.0]
    }
}

impl ops::Mul<&Matrix4x4Simd> for &Matrix4x4Simd {
    type Output = Matrix4x4Simd;

    fn mul(self, rhs: &Matrix4x4Simd) -> Self::Output {
        assert!(
            std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
        );
        // SAFETY: we've checked that all features are supported
        unsafe { self.multiply(rhs) }
    }
}

impl approx::AbsDiffEq for Matrix4x4Simd {
    type Epsilon = <f32 as AbsDiffEq>::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.flat_cells()
            .iter()
            .zip(other.flat_cells().iter())
            .all(|(lhs, rhs)| lhs.abs_diff_eq(rhs, epsilon))
    }
}

impl approx::RelativeEq for Matrix4x4Simd {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.flat_cells()
            .iter()
            .zip(other.flat_cells().iter())
            .all(|(lhs, rhs)| lhs.relative_eq(rhs, epsilon, max_relative))
    }
}

impl approx::UlpsEq for Matrix4x4Simd {
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.flat_cells()
            .iter()
            .zip(other.flat_cells().iter())
            .all(|(lhs, rhs)| lhs.ulps_eq(rhs, epsilon, max_ulps))
    }
}

impl Debug for Matrix4x4Simd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Matrix4x4Simd([")?;
        let mut first_row_written = false;
        for row in self.0.iter() {
            if f.alternate() {
                if !first_row_written {
                    f.write_str("\n")?
                }
                write!(f, "    {:?}", row)?;
                f.write_str(",\n")?;
            } else {
                if first_row_written {
                    f.write_str(", ")?
                }
                row.fmt(f)?;
            }
            first_row_written = true;
        }
        f.write_str("])")?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, assert_relative_eq, assert_ulps_eq};

    use super::*;

    #[test]
    fn test_avx_sse_supported() {
        assert!(std::arch::is_x86_feature_detected!("avx"));
        assert!(std::arch::is_x86_feature_detected!("avx2"));
        assert!(std::arch::is_x86_feature_detected!("fma"));
    }

    #[test]
    fn test_approx() {
        let matrix = Matrix4x4Simd::from_rows([
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
            [0., 0., 0., 4.],
        ]);

        // Introduce floating point error.
        let modified = matrix
            .clone()
            .map(|cell| (cell * 10000. + 3.) / 10000. - 3. / 10000.);

        assert_ne!(matrix, modified);
        assert_abs_diff_eq!(matrix, modified);
        assert_relative_eq!(matrix, modified);
        assert_ulps_eq!(matrix, modified);
    }

    #[test]
    fn test_multiply_identity() {
        let matrix = Matrix4x4Simd::from_rows([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);

        assert_abs_diff_eq!(matrix, &matrix * &Matrix4x4Simd::IDENTITY);
    }

    #[test]
    fn test_multiply() {
        let a = Matrix4x4Simd::from_rows([
            [1., 2., 0., 1.],
            [0., 1., 3., 2.],
            [4., 0., 1., 0.],
            [2., 1., 0., 1.],
        ]);
        let b = Matrix4x4Simd::from_rows([
            [2., 1., 3., 0.],
            [1., 0., 2., 1.],
            [0., 1., 1., 2.],
            [3., 0., 0., 1.],
        ]);
        let expected = Matrix4x4Simd::from_rows([
            [7., 1., 7., 3.],
            [7., 3., 5., 9.],
            [8., 5., 13., 2.],
            [8., 2., 8., 2.],
        ]);

        assert_abs_diff_eq!(expected, &a * &b);
    }
}
