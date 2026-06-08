use crate::api::types::checkpoint::Checkpoint;
use crate::api::types::model_summary::ModelSummary;

/// Save a model checkpoint to disk.
pub fn save_checkpoint<P: AsRef<std::path::Path>>(
    model: &dyn mllayers::layer::Layer,
    path: P,
    epoch: usize,
    best_val_loss: f32,
) -> mlautograd::MlResult<()> {
    Checkpoint::create_and_save(model, path, epoch, best_val_loss)
}

/// Load a checkpoint from disk and apply it to a model.
pub fn load_checkpoint<P: AsRef<std::path::Path>>(
    model: &mut dyn mllayers::layer::Layer,
    path: P,
) -> mlautograd::MlResult<Checkpoint> {
    Checkpoint::load_and_apply(model, path)
}

/// Produce a human-readable summary of a model's parameter budget.
pub fn model_summary(model: &dyn mllayers::layer::Layer) -> String {
    ModelSummary::summarize(model)
}
