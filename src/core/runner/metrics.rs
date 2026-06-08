use crate::api::types::metrics::Metrics;
use crate::api::traits::metrics_ops::MetricsOps;

impl MetricsOps for Metrics {
    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        assert_eq!(predictions.len(), targets.len());
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let p_f64 = *p as f64;
            let t_f64 = *t as f64;
            let diff = p_f64 - t_f64;
            self.sum_squared_error += diff * diff;
            self.sum_absolute_error += diff.abs();
            self.sum_targets += t_f64;
            self.sum_targets_squared += t_f64 * t_f64;
            self.sum_predictions += p_f64;
            self.count += 1;

            if *t != 0.0 {
                self.sum_mape += diff.abs() / t_f64.abs();
                self.mape_count += 1;
            }

            let denom = p_f64.abs() + t_f64.abs();
            if denom != 0.0 {
                self.sum_smape += 2.0 * diff.abs() / denom;
            }
        }
    }

    fn mse(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.sum_squared_error / self.count as f64
    }

    fn mae(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.sum_absolute_error / self.count as f64
    }

    fn rmse(&self) -> f64 { self.mse().sqrt() }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.sum_absolute_error = 0.0;
        self.count = 0;
        self.sum_targets = 0.0;
        self.sum_targets_squared = 0.0;
        self.sum_predictions = 0.0;
        self.sum_mape = 0.0;
        self.sum_smape = 0.0;
        self.mape_count = 0;
    }

    fn r_squared(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let ss_tot = self.sum_targets_squared
            - (self.sum_targets * self.sum_targets) / self.count as f64;
        if ss_tot == 0.0 { return 0.0; }
        1.0 - self.sum_squared_error / ss_tot
    }

    fn mape(&self) -> f64 {
        if self.mape_count == 0 { return 0.0; }
        (self.sum_mape / self.mape_count as f64) * 100.0
    }

    fn smape(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        (self.sum_smape / self.count as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: update
    #[test]
    fn test_metrics_mse_known_value() {
        let mut m = Metrics::new();
        m.update(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.mse() - 4.0).abs() < 1e-9);
    }

    /// @covers: update
    #[test]
    fn test_metrics_mae_known_value() {
        let mut m = Metrics::new();
        m.update(&[1.0, 5.0], &[3.0, 3.0]);
        assert!((m.mae() - 2.0).abs() < 1e-9);
    }

    /// @covers: rmse
    #[test]
    fn test_metrics_rmse_equals_sqrt_mse() {
        let mut m = Metrics::new();
        m.update(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.rmse() - m.mse().sqrt()).abs() < 1e-12);
    }

    /// @covers: reset
    #[test]
    fn test_metrics_reset_clears_state() {
        let mut m = Metrics::new();
        m.update(&[1.0], &[2.0]);
        m.reset();
        assert_eq!(m.mse(), 0.0);
        assert_eq!(m.mae(), 0.0);
    }

    /// @covers: r_squared
    #[test]
    fn test_metrics_r_squared_perfect_prediction() {
        let mut m = Metrics::new();
        m.update(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((m.r_squared() - 1.0).abs() < 1e-9);
    }

    /// @covers: mape
    #[test]
    fn test_metrics_mape_known_value() {
        let mut m = Metrics::new();
        // 50% error on a target of 2.0
        m.update(&[3.0], &[2.0]);
        assert!((m.mape() - 50.0).abs() < 1e-6);
    }

    /// @covers: smape
    #[test]
    fn test_metrics_smape_symmetric() {
        let mut m1 = Metrics::new();
        let mut m2 = Metrics::new();
        m1.update(&[3.0], &[1.0]);
        m2.update(&[1.0], &[3.0]);
        assert!((m1.smape() - m2.smape()).abs() < 1e-6);
    }
}
