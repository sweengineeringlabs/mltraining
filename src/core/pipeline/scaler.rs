/// Feature scaling / normalization for time series data.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalerType {
    MinMax,
    Standard,
    Robust,
}

#[derive(Debug, Clone)]
pub struct Scaler {
    scaler_type: ScalerType,
    params: Vec<(f32, f32)>,
}

impl Scaler {
    pub fn fit(data: &[Vec<f32>], scaler_type: ScalerType) -> Self {
        let params: Vec<(f32, f32)> = data.iter().map(|col| match scaler_type {
            ScalerType::MinMax => {
                let min = col.iter().copied().fold(f32::INFINITY, f32::min);
                let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                (min, if range == 0.0 { 1.0 } else { range })
            }
            ScalerType::Standard => {
                let n = col.len() as f32;
                if n == 0.0 { return (0.0, 1.0); }
                let mean = col.iter().sum::<f32>() / n;
                let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let std = variance.sqrt();
                (mean, if std == 0.0 { 1.0 } else { std })
            }
            ScalerType::Robust => {
                if col.is_empty() { return (0.0, 1.0); }
                let median = compute_median(col);
                let q25 = compute_percentile(col, 0.25);
                let q75 = compute_percentile(col, 0.75);
                let iqr = q75 - q25;
                (median, if iqr == 0.0 { 1.0 } else { iqr })
            }
        }).collect();
        Self { scaler_type, params }
    }

    pub fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter().zip(self.params.iter()).map(|(col, &(p1, p2))| {
            col.iter().map(|&x| (x - p1) / p2).collect()
        }).collect()
    }

    pub fn inverse_transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter().zip(self.params.iter()).map(|(col, &(p1, p2))| {
            col.iter().map(|&x| x * p2 + p1).collect()
        }).collect()
    }

    pub fn scaler_type(&self) -> ScalerType { self.scaler_type }
    pub fn params(&self) -> &[(f32, f32)] { &self.params }
}

fn compute_median(values: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 } else { sorted[n / 2] }
}

fn compute_percentile(values: &[f32], percentile: f32) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let idx = percentile * (n - 1) as f32;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f32;
    if lower == upper { sorted[lower] } else { sorted[lower] * (1.0 - frac) + sorted[upper] * frac }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_fit_transform() {
        let data = vec![vec![0.0, 10.0, 20.0, 30.0, 40.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        assert!((transformed[0][0] - 0.0).abs() < 1e-6);
        assert!((transformed[0][4] - 1.0).abs() < 1e-6);
        assert!((transformed[0][2] - 0.5).abs() < 1e-6);
    }

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
