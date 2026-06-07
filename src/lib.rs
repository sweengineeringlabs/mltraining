pub mod loss;
pub mod dataset;
pub mod lossfunction;
pub mod pipeline;
pub mod runner;
pub mod checkpoint;

pub use loss::Loss;
pub use dataset::Dataset;
pub use lossfunction::mse_loss::MSELoss;
pub use lossfunction::mae_loss::MAELoss;
pub use lossfunction::huber::HuberLoss;
pub use lossfunction::cross_entropy::CrossEntropyLoss;
pub use lossfunction::quantile::QuantileLoss;
pub use pipeline::dataloader::DataLoader;
pub use pipeline::scaler::{Scaler, ScalerType};
pub use runner::trainer::Trainer;
pub use runner::metrics::Metrics;
pub use runner::summary::model_summary;
pub use checkpoint::{Checkpoint, SavedParam, save_checkpoint, load_checkpoint};

// Re-exports from sub-crates so downstream crates need only depend on mltraining.
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
