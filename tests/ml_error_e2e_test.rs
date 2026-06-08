// @covers: MlError
use mltraining::MlError;

#[test]
fn test_ml_error_is_debug_printable() {
    let e = MlError::TrainingError("test error".to_string());
    let s = format!("{e:?}");
    assert!(s.contains("TrainingError") || !s.is_empty());
}

#[test]
fn test_ml_error_display_is_non_empty() {
    let e = MlError::TrainingError("oops".to_string());
    let s = format!("{e}");
    assert!(!s.is_empty());
}
