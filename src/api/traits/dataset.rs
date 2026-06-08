use mlautograd::{MlResult, Tensor};

/// A dataset that produces indexed (input, target) tensor pairs.
///
/// Implement this trait for any data source to use it with `DataLoader`.
pub trait Dataset {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the (input, target) pair at the given index.
    fn get(&self, index: usize) -> MlResult<(Tensor, Tensor)>;

    /// Shape of a single input sample (e.g., [window_size, num_features]).
    fn input_shape(&self) -> Vec<usize>;

    /// Dimensionality of the target (e.g., 1 for single-target regression).
    fn target_dim(&self) -> usize;
}
