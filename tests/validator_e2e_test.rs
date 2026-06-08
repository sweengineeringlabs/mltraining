// @covers: Validator::validate
use mltraining::{Validator, MlResult, MlError};

/// A test validator that rejects empty strings.
struct NonEmptyStringValidator;

impl Validator for NonEmptyStringValidator {
    type Target = String;

    fn validate(&self, target: &String) -> MlResult<()> {
        if target.is_empty() {
            Err(MlError::InvalidConfig("string must not be empty".into()))
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_validator_accepts_valid_input() {
    let v = NonEmptyStringValidator;
    assert!(v.validate(&"hello".to_string()).is_ok());
}

#[test]
fn test_validator_rejects_invalid_input() {
    let v = NonEmptyStringValidator;
    assert!(v.validate(&String::new()).is_err());
}
