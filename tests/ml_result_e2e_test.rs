// @covers: MlResult
use mltraining::{MlResult, MlError};

fn ok_result() -> MlResult<i32> { Ok(42) }
fn err_result() -> MlResult<i32> { Err(MlError::TrainingError("fail".to_string())) }

#[test]
fn test_ml_result_ok_unwraps_correctly() {
    assert_eq!(ok_result().expect("ok"), 42);
}

#[test]
fn test_ml_result_err_is_error() {
    assert!(err_result().is_err());
}

#[test]
fn test_ml_result_map_applies_transform() {
    let result: MlResult<i32> = Ok(10);
    let doubled = result.map(|v| v * 2).expect("map");
    assert_eq!(doubled, 20);
}
