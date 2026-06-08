use mlautograd::MlResult;
use mllayers::layer::Layer;

/// Operations for saving and loading model checkpoints.
pub trait CheckpointOps {
    /// Persist this checkpoint to disk at the given path.
    fn save_to<P: AsRef<std::path::Path>>(&self, path: P) -> MlResult<()>;

    /// Load this checkpoint's parameters into the given model.
    fn load_into_model(&self, model: &mut dyn Layer) -> MlResult<()>;
}
