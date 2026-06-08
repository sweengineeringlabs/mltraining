// @covers: ScaleTransform::transform, ScaleTransform::inverse_transform,
//          ScaleTransform::scaler_type, ScaleTransform::params
use mltraining::{Scaler, ScalerType, ScaleTransform};

#[test]
fn test_scale_transform_minmax_round_trip() {
    let data = vec![vec![0.0, 5.0, 10.0]];
    let scaler = Scaler::fit(&data, ScalerType::MinMax);
    let transformed = scaler.transform(&data);
    let recovered = scaler.inverse_transform(&transformed);
    for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
        assert!((orig - rec).abs() < 1e-5, "round-trip failed: {orig} != {rec}");
    }
}

#[test]
fn test_scale_transform_scaler_type_returns_fitted_type() {
    let data = vec![vec![1.0, 2.0, 3.0]];
    let scaler = Scaler::fit(&data, ScalerType::Standard);
    assert_eq!(scaler.scaler_type(), ScalerType::Standard);
}

#[test]
fn test_scale_transform_params_has_one_entry_per_column() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let scaler = Scaler::fit(&data, ScalerType::MinMax);
    assert_eq!(scaler.params().len(), 2);
}

#[test]
fn test_scale_transform_robust_is_stable_on_constant_column() {
    let data = vec![vec![7.0, 7.0, 7.0]];
    let scaler = Scaler::fit(&data, ScalerType::Robust);
    let transformed = scaler.transform(&data);
    for &v in &transformed[0] {
        assert!(v.is_finite(), "non-finite value in robust transform of constant column");
    }
}
