// @covers: ScalerType
use mltraining::ScalerType;

#[test]
fn test_scaler_type_variants_are_distinct() {
    assert_ne!(ScalerType::MinMax, ScalerType::Standard);
    assert_ne!(ScalerType::Standard, ScalerType::Robust);
    assert_ne!(ScalerType::MinMax, ScalerType::Robust);
}

#[test]
fn test_scaler_type_is_copy() {
    let t = ScalerType::MinMax;
    let t2 = t;
    assert_eq!(t, t2);
}

#[test]
fn test_scaler_type_debug_is_non_empty() {
    let s = format!("{:?}", ScalerType::Standard);
    assert!(!s.is_empty());
}
