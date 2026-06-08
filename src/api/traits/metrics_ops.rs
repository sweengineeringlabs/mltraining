/// Accumulated regression metrics operations.
pub trait MetricsOps {
    /// Accumulate predictions and targets into running totals.
    fn update(&mut self, predictions: &[f32], targets: &[f32]);

    /// Mean Squared Error over accumulated samples.
    fn mse(&self) -> f64;

    /// Mean Absolute Error over accumulated samples.
    fn mae(&self) -> f64;

    /// Root Mean Squared Error.
    fn rmse(&self) -> f64;

    /// Reset all accumulators to zero.
    fn reset(&mut self);

    /// Coefficient of determination (R²).
    fn r_squared(&self) -> f64;

    /// Mean Absolute Percentage Error (%).
    fn mape(&self) -> f64;

    /// Symmetric Mean Absolute Percentage Error (%).
    fn smape(&self) -> f64;
}
