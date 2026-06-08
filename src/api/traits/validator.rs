use mlautograd::MlResult;

/// Validates training configuration and data inputs before a training run.
pub trait Validator {
    /// The type being validated.
    type Target;

    /// Validate the target, returning an error if it fails validation.
    fn validate(&self, target: &Self::Target) -> MlResult<()>;
}
