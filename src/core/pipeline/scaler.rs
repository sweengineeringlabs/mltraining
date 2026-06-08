use crate::api::types::scaler::Scaler;
use crate::api::types::scaler_type::ScalerType;
use crate::api::traits::scale_transform::ScaleTransform;

impl ScaleTransform for Scaler {
    fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter().zip(self.params.iter()).map(|(col, &(p1, p2))| {
            col.iter().map(|&x| (x - p1) / p2).collect()
        }).collect()
    }

    fn inverse_transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter().zip(self.params.iter()).map(|(col, &(p1, p2))| {
            col.iter().map(|&x| x * p2 + p1).collect()
        }).collect()
    }

    fn scaler_type(&self) -> ScalerType {
        self.scaler_type
    }

    fn params(&self) -> &[(f32, f32)] {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: transform
    #[test]
    fn test_minmax_fit_transform() {
        let data = vec![vec![0.0, 10.0, 20.0, 30.0, 40.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        assert!((transformed[0][0] - 0.0).abs() < 1e-6);
        assert!((transformed[0][4] - 1.0).abs() < 1e-6);
        assert!((transformed[0][2] - 0.5).abs() < 1e-6);
    }

    /// @covers: inverse_transform
    #[test]
    fn test_minmax_inverse_transform() {
        let data = vec![vec![0.0, 10.0, 20.0, 30.0, 40.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        let recovered = scaler.inverse_transform(&transformed);
        for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
            assert!((orig - rec).abs() < 1e-5);
        }
    }

    /// @covers: transform
    #[test]
    fn test_standard_fit_transform() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let scaler = Scaler::fit(&data, ScalerType::Standard);
        let transformed = scaler.transform(&data);
        let mean: f32 = transformed[0].iter().sum::<f32>() / transformed[0].len() as f32;
        assert!(mean.abs() < 1e-5);
        let var: f32 = transformed[0].iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / transformed[0].len() as f32;
        assert!((var.sqrt() - 1.0).abs() < 1e-5);
    }

    /// @covers: transform
    #[test]
    fn test_constant_column_handling() {
        let data = vec![vec![5.0, 5.0, 5.0, 5.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] { assert!(v.is_finite()); }
        let scaler = Scaler::fit(&data, ScalerType::Standard);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] { assert!(v.is_finite()); }
        let scaler = Scaler::fit(&data, ScalerType::Robust);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] { assert!(v.is_finite()); }
    }
}
