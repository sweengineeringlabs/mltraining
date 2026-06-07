pub struct Metrics {
    sum_squared_error: f64,
    sum_absolute_error: f64,
    count: usize,
    sum_targets: f64,
    sum_targets_squared: f64,
    sum_predictions: f64,
    sum_mape: f64,
    sum_smape: f64,
    mape_count: usize,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            sum_absolute_error: 0.0,
            count: 0,
            sum_targets: 0.0,
            sum_targets_squared: 0.0,
            sum_predictions: 0.0,
            sum_mape: 0.0,
            sum_smape: 0.0,
            mape_count: 0,
        }
    }

    pub fn update(&mut self, predictions: &[f32], targets: &[f32]) {
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

    pub fn mse(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.sum_squared_error / self.count as f64
    }

    pub fn mae(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.sum_absolute_error / self.count as f64
    }

    pub fn rmse(&self) -> f64 { self.mse().sqrt() }

    pub fn reset(&mut self) {
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

    pub fn r_squared(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let ss_tot = self.sum_targets_squared
            - (self.sum_targets * self.sum_targets) / self.count as f64;
        if ss_tot == 0.0 { return 0.0; }
        1.0 - self.sum_squared_error / ss_tot
    }

    pub fn mape(&self) -> f64 {
        if self.mape_count == 0 { return 0.0; }
        (self.sum_mape / self.mape_count as f64) * 100.0
    }

    pub fn smape(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        (self.sum_smape / self.count as f64) * 100.0
    }
}

impl Default for Metrics {
    fn default() -> Self { Self::new() }
}
