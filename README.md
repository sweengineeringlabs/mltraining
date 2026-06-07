# mltraining

End-to-end ML training infrastructure for Rust — loss functions, data pipeline, trainer loop with early stopping and checkpointing, evaluation metrics.

Sits on top of `mlautograd`, `mllayers`, and `mloptim`. All error handling uses `mlautograd::MlError` / `MlResult`.

## Use cases

### Supervised regression: end-to-end training loop
Wire a model, optimizer, and loss function into a `Trainer` and call `fit()` for the full loop.

```rust
use mltraining::{Trainer, MSELoss};
use mllayers::{Linear, ReLU, Sequential};
use mloptim::AdamW;
use mlautograd::Tensor;

let model = Sequential::new(vec![
    Box::new(Linear::new(10, 32)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(32, 1)),
]);
let optimizer = AdamW::new(1e-3);
let loss_fn = MSELoss::new();

let mut trainer = Trainer::new(model, optimizer, loss_fn)
    .with_grad_clip(1.0)
    .with_early_stopping(5)
    .with_checkpoint_dir("checkpoints");

let history = trainer.fit(&train_batches, &val_batches, 100).unwrap();
```

### Time-series forecasting: data pipeline
Use `Scaler` to normalize features and `DataLoader` to batch your dataset.

```rust
use mltraining::{Scaler, ScalerType, DataLoader};

let scaler = Scaler::fit(&feature_columns, ScalerType::Standard);
let normalized = scaler.transform(&feature_columns);
// ... build dataset from normalized data ...
let loader = DataLoader::new(dataset, batch_size: 64, shuffle: true);
for (inputs, targets) in &mut loader { /* train */ }
```

### Classification: cross-entropy with quantile analysis
Train a classifier and evaluate calibration with quantile loss.

```rust
use mltraining::{CrossEntropyLoss, QuantileLoss};

let ce_loss = CrossEntropyLoss::new(); // expects one-hot targets
let q90_loss = QuantileLoss::new(0.9); // 90th percentile loss for upper bound
```

### Model evaluation: metrics and summary
Track MSE, MAE, RMSE, R², MAPE, SMAPE per epoch and inspect the parameter budget.

```rust
use mltraining::{Metrics, model_summary};

let mut metrics = Metrics::new();
metrics.update(&predictions, &targets);
println!("RMSE: {:.4}, R²: {:.4}", metrics.rmse(), metrics.r_squared());
println!("{}", model_summary(&model));
```

### Checkpointing: save and restore best model
The trainer automatically saves a checkpoint when validation loss improves.
Load it back for inference or continued training.

```rust
use mltraining::load_checkpoint;

let checkpoint = load_checkpoint(&mut model, "checkpoints/best_model.bin").unwrap();
println!("Loaded epoch {} (val_loss={:.6})", checkpoint.epoch, checkpoint.best_val_loss);
```

## Crate layout

| Module | Contents |
|---|---|
| `loss` | `Loss` trait |
| `dataset` | `Dataset` trait |
| `lossfunction::mse_loss` | `MSELoss` |
| `lossfunction::mae_loss` | `MAELoss` |
| `lossfunction::huber` | `HuberLoss` |
| `lossfunction::cross_entropy` | `CrossEntropyLoss` |
| `lossfunction::quantile` | `QuantileLoss` |
| `pipeline::dataloader` | `DataLoader<D>` (batched iterator) |
| `pipeline::scaler` | `Scaler`, `ScalerType` (MinMax / Standard / Robust) |
| `runner::trainer` | `Trainer` (fit, train_epoch, validate, predict) |
| `runner::metrics` | `Metrics` (MSE, MAE, RMSE, R², MAPE, SMAPE) |
| `runner::summary` | `model_summary` |
| `checkpoint` | `Checkpoint`, `save_checkpoint`, `load_checkpoint` |
