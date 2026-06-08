// @covers: Scaler
use mltraining::{Scaler, ScalerType, ScaleTransform};

#[test]
fn test_scaler_minmax_fit_normalises_to_unit_range() {
    let data = vec![vec![0.0, 5.0, 10.0]];
    let scaler = Scaler::fit(&data, ScalerType::MinMax);
    let t = scaler.transform(&data);
    assert!((t[0][0] - 0.0).abs() < 1e-6);
    assert!((t[0][1] - 0.5).abs() < 1e-6);
    assert!((t[0][2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaler_inverse_transform_recovers_original() {
    let data = vec![vec![1.0, 3.0, 5.0, 7.0]];
    let scaler = Scaler::fit(&data, ScalerType::Standard);
    let transformed = scaler.transform(&data);
    let recovered = scaler.inverse_transform(&transformed);
    for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
        assert!((orig - rec).abs() < 1e-5, "orig={orig}, rec={rec}");
    }
}

#[test]
fn test_scaler_type_is_preserved_after_fit() {
    let data = vec![vec![1.0, 2.0, 3.0]];
    let scaler = Scaler::fit(&data, ScalerType::Robust);
    assert_eq!(scaler.scaler_type(), ScalerType::Robust);
}
