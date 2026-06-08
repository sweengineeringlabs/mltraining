use crate::api::types::scaler_type::ScalerType;

/// Feature scaling and inverse-scaling operations.
pub trait ScaleTransform {
    /// Transform the data using the fitted parameters.
    fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Reverse the transform to recover original-scale values.
    fn inverse_transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// The kind of scaler that was fitted.
    fn scaler_type(&self) -> ScalerType;

    /// Fitted (offset, scale) parameters per feature column.
    fn params(&self) -> &[(f32, f32)];
}
