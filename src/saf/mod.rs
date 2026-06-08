// mltraining public API
// saf/ re-exports from api/ only; free-fn wrappers are defined in saf/functions.rs.
mod mltraining_svc;

// Types
pub use crate::api::types::checkpoint::Checkpoint;
pub use crate::api::types::saved_param::SavedParam;
pub use crate::api::types::cross_entropy_loss::CrossEntropyLoss;
pub use crate::api::types::huber_loss::HuberLoss;
pub use crate::api::types::m_a_e_loss::MAELoss;
pub use crate::api::types::m_s_e_loss::MSELoss;
pub use crate::api::types::quantile_loss::QuantileLoss;
pub use crate::api::types::data_loader::DataLoader;
pub use crate::api::types::scaler::Scaler;
pub use crate::api::types::scaler_type::ScalerType;
pub use crate::api::types::trainer::Trainer;
pub use crate::api::types::metrics::Metrics;
pub use crate::api::types::model_summary::ModelSummary;

// Traits — re-exported via domain api/ submodules.
pub use crate::api::lossfunction::Loss;
pub use crate::api::pipeline::Dataset;
pub use crate::api::checkpoint::CheckpointOps;
pub use crate::api::runner::MetricsOps;
pub use crate::api::runner::TrainerOps;
pub use crate::api::runner::Validator;
pub use crate::api::pipeline::ScaleTransform;

// Convenience free functions (re-exported from saf/functions.rs).
pub use mltraining_svc::{save_checkpoint, load_checkpoint, model_summary};

// umbrella re-exports (mltraining is the single entry point for the ml* stack)
pub use mlautograd::{MlError, MlResult, Tensor, TensorId, pool, tape};
pub use mlautograd::{BackwardOp, TapeEntry, unbroadcast, AddBackward, MatMulBackward, MulBackward, ReLUBackward};
pub use mllayers::{
    Layer,
    Linear, Conv1d, Conv1dBuilder, BatchNorm1d, BatchNorm1dBuilder,
    Dropout, LayerNorm, Sequential,
    GELU, ReLU, SiLU, Sigmoid, Tanh,
};
pub use mloptim::{Optimizer, LRScheduler, Adam, AdamW, SGD, StepLR, CosineAnnealingLR, WarmupCosineScheduler};
pub use mloptim::{clip_grad_norm, clip_grad_value};
