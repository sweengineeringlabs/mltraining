// @covers: MetricsOps::update, MetricsOps::mse, MetricsOps::mae, MetricsOps::rmse,
//          MetricsOps::reset, MetricsOps::r_squared, MetricsOps::mape, MetricsOps::smape
use mltraining::{Metrics, MetricsOps};

#[test]
fn test_metrics_ops_update_and_mse() {
    let mut m = Metrics::new();
    m.update(&[3.0], &[1.0]);
    // (3-1)^2 = 4, mse = 4
    assert!((m.mse() - 4.0).abs() < 1e-9, "mse={}", m.mse());
}

#[test]
fn test_metrics_ops_mae_known_value() {
    let mut m = Metrics::new();
    m.update(&[5.0, 1.0], &[3.0, 3.0]);
    assert!((m.mae() - 2.0).abs() < 1e-9, "mae={}", m.mae());
}

#[test]
fn test_metrics_ops_rmse_equals_sqrt_mse() {
    let mut m = Metrics::new();
    m.update(&[0.0, 4.0], &[2.0, 2.0]);
    assert!((m.rmse() - m.mse().sqrt()).abs() < 1e-12);
}

#[test]
fn test_metrics_ops_reset_clears_state() {
    let mut m = Metrics::new();
    m.update(&[5.0], &[1.0]);
    m.reset();
    assert_eq!(m.mse(), 0.0);
}

#[test]
fn test_metrics_ops_r_squared_perfect() {
    let mut m = Metrics::new();
    m.update(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((m.r_squared() - 1.0).abs() < 1e-9);
}

#[test]
fn test_metrics_ops_mape() {
    let mut m = Metrics::new();
    m.update(&[3.0], &[2.0]);
    assert!((m.mape() - 50.0).abs() < 1e-6);
}

#[test]
fn test_metrics_ops_smape_symmetric() {
    let mut m1 = Metrics::new();
    let mut m2 = Metrics::new();
    m1.update(&[3.0], &[1.0]);
    m2.update(&[1.0], &[3.0]);
    assert!((m1.smape() - m2.smape()).abs() < 1e-6);
}
