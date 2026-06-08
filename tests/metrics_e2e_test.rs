// @covers: Metrics::update, Metrics::mse, Metrics::mae, Metrics::rmse,
//          Metrics::reset, Metrics::r_squared, Metrics::mape, Metrics::smape
use mltraining::{Metrics, MetricsOps};

#[test]
fn test_metrics_mse_known_value() {
    let mut m = Metrics::new();
    m.update(&[1.0, 2.0], &[3.0, 4.0]);
    // errors are -2 and -2, so mse = (4+4)/2 = 4
    assert!((m.mse() - 4.0).abs() < 1e-9, "mse={}", m.mse());
}

#[test]
fn test_metrics_mae_known_value() {
    let mut m = Metrics::new();
    m.update(&[1.0, 5.0], &[3.0, 3.0]);
    // errors are |1-3|=2, |5-3|=2, mae=2
    assert!((m.mae() - 2.0).abs() < 1e-9, "mae={}", m.mae());
}

#[test]
fn test_metrics_rmse_equals_sqrt_mse() {
    let mut m = Metrics::new();
    m.update(&[0.0, 4.0], &[2.0, 2.0]);
    assert!((m.rmse() - m.mse().sqrt()).abs() < 1e-12);
}

#[test]
fn test_metrics_reset_clears_all_accumulators() {
    let mut m = Metrics::new();
    m.update(&[5.0], &[1.0]);
    m.reset();
    assert_eq!(m.mse(), 0.0);
    assert_eq!(m.mae(), 0.0);
    assert_eq!(m.rmse(), 0.0);
    assert_eq!(m.r_squared(), 0.0);
    assert_eq!(m.mape(), 0.0);
    assert_eq!(m.smape(), 0.0);
}

#[test]
fn test_metrics_r_squared_perfect_prediction() {
    let mut m = Metrics::new();
    m.update(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((m.r_squared() - 1.0).abs() < 1e-9, "r2={}", m.r_squared());
}

#[test]
fn test_metrics_mape_known_value() {
    let mut m = Metrics::new();
    // pred=3, tgt=2 → error=1, mape = 1/2 * 100 = 50%
    m.update(&[3.0], &[2.0]);
    assert!((m.mape() - 50.0).abs() < 1e-6, "mape={}", m.mape());
}

#[test]
fn test_metrics_smape_symmetric() {
    let mut m1 = Metrics::new();
    let mut m2 = Metrics::new();
    m1.update(&[3.0], &[1.0]);
    m2.update(&[1.0], &[3.0]);
    assert!((m1.smape() - m2.smape()).abs() < 1e-6);
}

#[test]
fn test_metrics_mape_ignores_zero_targets() {
    let mut m = Metrics::new();
    // target=0 should be excluded from MAPE
    m.update(&[1.0], &[0.0]);
    assert_eq!(m.mape(), 0.0, "mape should be 0 when all targets are 0");
}
