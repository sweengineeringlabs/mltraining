/// Accumulated regression metrics.
pub struct Metrics {
    pub(crate) sum_squared_error: f64,
    pub(crate) sum_absolute_error: f64,
    pub(crate) count: usize,
    pub(crate) sum_targets: f64,
    pub(crate) sum_targets_squared: f64,
    pub(crate) sum_predictions: f64,
    pub(crate) sum_mape: f64,
    pub(crate) sum_smape: f64,
    pub(crate) mape_count: usize,
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
}

impl Default for Metrics {
    fn default() -> Self { Metrics::new() }
}
