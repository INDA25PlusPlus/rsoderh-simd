use std::ops;

use approx::AbsDiffEq;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix4x4Sisd(pub [[f32; 4]; 4]);

impl Matrix4x4Sisd {
    pub const ZERO: Self = Self([[0.; 4]; 4]);

    pub const IDENTITY: Self = Self([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]);

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

    pub fn flat_cells(&self) -> &[f32; 4 * 4] {
        bytemuck::cast_ref(&self.0)
    }

    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        Self(self.0.map(|row| row.map(&f)))
    }
}

impl ops::Index<(usize, usize)> for Matrix4x4Sisd {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.1][index.0]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix4x4Sisd {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.1][index.0]
    }
}

impl ops::Mul<&Matrix4x4Sisd> for &Matrix4x4Sisd {
    type Output = Matrix4x4Sisd;
    fn mul(self, rhs: &Matrix4x4Sisd) -> Self::Output {
        let mut result = Matrix4x4Sisd::ZERO;
        for (row, row_cells) in result.0.iter_mut().enumerate() {
            for (column, cell) in row_cells.iter_mut().enumerate() {
                *cell = (0..4)
                    .map(|column| self[(column, row)])
                    .zip((0..4).map(|row| rhs[(column, row)]))
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }

        result
    }
}

impl approx::AbsDiffEq for Matrix4x4Sisd {
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

impl approx::RelativeEq for Matrix4x4Sisd {
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

impl approx::UlpsEq for Matrix4x4Sisd {
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

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, assert_relative_eq, assert_ulps_eq};

    use super::*;

    #[test]
    fn test_approx() {
        let matrix = Matrix4x4Sisd::from_rows([
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
        let matrix = Matrix4x4Sisd::from_rows([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);

        assert_abs_diff_eq!(matrix, &matrix * &Matrix4x4Sisd::IDENTITY);
    }

    #[test]
    fn test_multiply() {
        let a = Matrix4x4Sisd::from_rows([
            [1., 2., 0., 1.],
            [0., 1., 3., 2.],
            [4., 0., 1., 0.],
            [2., 1., 0., 1.],
        ]);
        let b = Matrix4x4Sisd::from_rows([
            [2., 1., 3., 0.],
            [1., 0., 2., 1.],
            [0., 1., 1., 2.],
            [3., 0., 0., 1.],
        ]);
        let expected = Matrix4x4Sisd::from_rows([
            [7., 1., 7., 3.],
            [7., 3., 5., 9.],
            [8., 5., 13., 2.],
            [8., 2., 8., 2.],
        ]);

        assert_abs_diff_eq!(expected, &a * &b);
    }
}
