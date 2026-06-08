use crate::api::types::scaler_type::ScalerType;

/// Fitted feature scaler for normalizing time series data.
#[derive(Debug, Clone)]
pub struct Scaler {
    pub(crate) scaler_type: ScalerType,
    pub(crate) params: Vec<(f32, f32)>,
}

impl Scaler {
    /// Fit a scaler to the given column-major data using the specified algorithm.
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
                let median = Self::compute_median(col);
                let q25 = Self::compute_percentile(col, 0.25);
                let q75 = Self::compute_percentile(col, 0.75);
                let iqr = q75 - q25;
                (median, if iqr == 0.0 { 1.0 } else { iqr })
            }
        }).collect();
        Self { scaler_type, params }
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
}
