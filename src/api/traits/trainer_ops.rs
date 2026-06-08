use mlautograd::{MlResult, Tensor};

/// Core training operations for a Trainer.
pub trait TrainerOps {
    /// Run a single training epoch over the provided batches.
    ///
    /// Returns the mean training loss for the epoch.
    fn train_epoch(&mut self, batches: &[(Tensor, Tensor)]) -> MlResult<f32>;

    /// Evaluate the model on validation batches without updating weights.
    ///
    /// Returns the mean validation loss.
    fn validate(&mut self, batches: &[(Tensor, Tensor)]) -> MlResult<f32>;

    /// Run the full training loop for `epochs` with optional early stopping
    /// and checkpointing.
    ///
    /// Returns the per-epoch `(train_loss, val_loss)` history.
    fn fit(
        &mut self,
        train_batches: &[(Tensor, Tensor)],
        val_batches: &[(Tensor, Tensor)],
        epochs: usize,
    ) -> MlResult<Vec<(f32, f32)>>;

    /// Run inference on a single input tensor.
    fn predict(&mut self, input: &Tensor) -> MlResult<Tensor>;
}
