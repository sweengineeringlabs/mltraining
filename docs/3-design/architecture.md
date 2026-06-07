# mltraining — Architecture

## Overview

`mltraining` is the umbrella crate for the ml* stack. It wires together `mlautograd` (automatic differentiation), `mllayers` (model building blocks), and `mloptim` (optimizers and schedulers) into a complete training system. On top of those re-exports it provides its own modules: differentiable loss functions, a data pipeline (`Dataset`, `DataLoader`, `Scaler`), the `Trainer` loop that drives the forward–loss–backward–step cycle, evaluation `Metrics`, and `Checkpoint` serialization. Downstream consumers — model architectures in llmeco and any application code — depend on `mltraining` as their single entry point for the entire ml* type surface.

---

## Stakeholders & Concerns

| Stakeholder | Role | Primary Concern |
|-------------|------|-----------------|
| Consumers | Model authors and application code that `use mltraining::*` | One import gives access to the entire ml* stack; training loop is correct and reproducible |
| Maintainers | Engineers extending loss functions, metrics, or the trainer | Trainer ownership model is clear; no lifetime tangles between model, optimizer, and data |
| Contributors | Adding new loss functions or pipeline stages | `Loss` and `Dataset` traits have minimal, stable contracts; module boundaries are clean |

---

## Component Diagram

```
mltraining  (SEA: api / core / saf)
│
├── lib.rs                        (declares api, core as private; re-exports saf::*)
│
├── saf/
│   └── mod.rs                    (sole public re-export surface)
│       ├── own types  → Loss, Dataset, MSELoss, MAELoss, HuberLoss,
│       │                  CrossEntropyLoss, QuantileLoss, DataLoader,
│       │                  Scaler, ScalerType, Trainer, Metrics,
│       │                  model_summary, Checkpoint, SavedParam,
│       │                  save_checkpoint, load_checkpoint
│       └── umbrella   → mlautograd, mllayers, mloptim (full public surface)
│
├── api/                          (public traits and value objects; no deps on core/)
│   ├── loss.rs                   [Loss trait]
│   │   └── forward(&Tensor, &Tensor) -> MlResult<Tensor>
│   └── dataset.rs                [Dataset trait]
│       └── len() / get(i) / input_shape() / target_dim()
│
└── core/                         (implementations; not re-exported from lib.rs directly)
    ├── lossfunction/
    │   ├── mse_loss.rs  → MSELoss
    │   ├── mae_loss.rs  → MAELoss
    │   ├── huber.rs     → HuberLoss         (delta as field)
    │   ├── cross_entropy.rs → CrossEntropyLoss
    │   └── quantile.rs  → QuantileLoss      (quantile as field)
    ├── pipeline/
    │   ├── dataloader.rs → DataLoader       (implements Iterator)
    │   └── scaler.rs     → Scaler, ScalerType (MinMax | Standard | Robust)
    ├── runner/
    │   ├── trainer.rs   → Trainer           (owns model + optimizer; train loop)
    │   ├── metrics.rs   → Metrics           (MAE, MSE, RMSE accumulator)
    │   └── summary.rs   → model_summary     (prints parameter counts)
    └── checkpoint/
        └── mod.rs       → Checkpoint, SavedParam, save_checkpoint, load_checkpoint
```

---

## Layer Responsibilities

| Layer | Module | Responsibility |
|-------|--------|---------------|
| facade | `lib.rs` | Declares `api` and `core` as private; re-exports `saf::*` to form the public surface |
| saf | `saf/mod.rs` | Sole public re-export surface; exposes mltraining types and the full mlautograd/mllayers/mloptim surface |
| api | `api/loss.rs` | Defines the `Loss` trait — one method, `forward`, returning a differentiable scalar tensor; no deps on `core/` |
| api | `api/dataset.rs` | Defines the `Dataset` trait — provides length, indexed access, shape metadata; no deps on `core/` |
| core | `core/lossfunction/*` | Implements concrete loss functions; stateless except for configuration fields (delta, quantile) |
| core | `core/pipeline/dataloader.rs` | Wraps a `Dataset`, implements `Iterator` to yield batches, optionally shuffles index order each epoch |
| core | `core/pipeline/scaler.rs` | Fits normalization statistics on training data; applies forward and inverse transforms |
| core | `core/runner/trainer.rs` | Owns model and optimizer; drives the `zero_grad → forward → loss → backward → step → scheduler` cycle |
| core | `core/runner/metrics.rs` | Accumulates per-batch predictions and targets; computes epoch-level regression metrics |
| core | `core/runner/summary.rs` | Walks `Layer::parameters()` and prints a human-readable parameter count table |
| core | `core/checkpoint/mod.rs` | Serializes `SavedParam` (flat `Vec<f32>` + shape) to disk; deserializes and restores into a model |

---

## Data Flow

```
Raw data (arrays / files)
        │
        ▼
  Scaler::fit(data)              ← compute mean/std or min/max on training split
  Scaler::transform(data)        ← normalize features in-place
        │
        ▼
  DataLoader::new(dataset, batch_size, shuffle)
        │
        ▼
  Trainer::train(epochs)
    ├─ DataLoader::next_batch()  ← yields (inputs: Tensor, targets: Tensor)
    ├─ model.forward(inputs)     ← Layer trait — uses mllayers
    ├─ loss_fn.forward(pred, tgt)← Loss trait — returns scalar Tensor
    ├─ tape::backward(loss)      ← mlautograd — fills .grad on all params
    ├─ clip_grad_norm (optional) ← mloptim free function
    ├─ optimizer.step(params)    ← mloptim — updates .data, clears .grad
    ├─ scheduler.step(&mut opt)  ← mloptim — adjusts LR
    └─ metrics.update(pred, tgt) ← accumulates batch stats
        │
        ▼
  save_checkpoint(model, path)   ← on improvement or end of training
        │
        ▼
  load_checkpoint(model, path)   ← restore for inference or resumed training
```

---

## Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant SC as Scaler
    participant DL as DataLoader
    participant TR as Trainer
    participant M as Model (Layer)
    participant L as Loss
    participant TP as tape (mlautograd)
    participant O as Optimizer
    participant S as LRScheduler
    participant CP as Checkpoint

    U->>SC: fit(train_data)
    SC-->>U: fitted Scaler
    U->>SC: transform(train_data)
    SC-->>U: normalized features

    U->>TR: train(epochs)

    loop each epoch
        loop each batch
            TR->>DL: next_batch()
            DL-->>TR: (inputs: Tensor[batch,seq,feat], targets: Tensor[batch,out])
            TR->>O: zero_grad(params)
            TR->>M: forward(&inputs)
            M-->>TR: pred: Tensor[batch,out]
            TR->>L: forward(&pred, &targets)
            L-->>TR: loss: Tensor (scalar)
            TR->>TP: tape::backward(&loss)
            TP-->>TR: grads populated on params
            TR->>O: step(params)
            TR->>S: step(&mut opt)
        end
        TR->>CP: save_checkpoint(model, path)
    end
    TR-->>U: trained model
```

## Dataflow Diagram

```mermaid
flowchart TD
    A["Raw Data<br/>Vec&lt;f32&gt; features"] --> B["Scaler::fit + transform<br/>IN: raw features<br/>OUT: normalized Vec&lt;f32&gt;"]
    B --> C["Dataset (impl Dataset trait)<br/>len() / get(i) → (Tensor, Tensor)"]
    C --> D["DataLoader::next_batch()<br/>IN: batch_size: usize, shuffle: bool<br/>OUT: (inputs: Tensor[batch,…], targets: Tensor[batch,out])"]
    D --> E["Model::forward<br/>IN: inputs: Tensor[batch,seq,feat]<br/>OUT: pred: Tensor[batch,out]"]
    E --> F["Loss::forward<br/>IN: pred: Tensor[batch,out]<br/>IN: targets: Tensor[batch,out]<br/>OUT: loss: Tensor (scalar)"]
    F --> G["tape::backward<br/>IN: loss: &Tensor<br/>OUT: grads on all requires_grad params"]
    G --> H["Optimizer::step<br/>IN: params: Vec&lt;&mut Tensor&gt;<br/>OUT: params updated in-place"]
    H --> I["Metrics::update<br/>IN: pred, targets<br/>OUT: running MAE/MSE/RMSE"]
    I --> J["Checkpoint::save<br/>IN: model params as Vec&lt;SavedParam&gt;<br/>OUT: .ckpt file on disk"]
    J --> K["Next Epoch"]
```

---

## Design Decisions

1. **`mltraining` is the umbrella crate** — downstream consumers depend on only one crate. All ml* public types are available via `use mltraining::*`, eliminating the need to manage separate version pins for mlautograd, mllayers, and mloptim.

2. **`Trainer` takes ownership of model and optimizer** — avoids lifetime tangles. The model is accessible via `trainer.model()` after training completes. Ownership ensures the training loop has exclusive access throughout.

3. **`Loss` trait operates on `&Tensor` pairs** — loss functions are stateless. Configuration (delta for Huber, quantile level for QuantileLoss) is stored as a field on the struct, not threaded through the trait method. This keeps the trait contract minimal and stable.

4. **`DataLoader` does not implement `Iterator`** — `next_batch()` is a method rather than a trait impl to avoid lifetime issues with the borrowed `Dataset`. The caller controls the loop.

5. **`Checkpoint` uses `SavedParam` (name + flat `Vec<f32>`)** — the serialization format is human-inspectable and framework-agnostic. It does not depend on any internal tensor representation, so checkpoints remain loadable if internal data structures change.

---

## Cross-Cutting Concerns

### Security
- Checkpoint files are deserialized as `Vec<SavedParam>` (name + flat `Vec<f32>`) — no executable code or unsafe deserialization
- No network I/O — all data movement is in-process or local filesystem
- Scaler statistics are computed from caller-provided training data; no untrusted external input paths

### Error Handling
- `Trainer::train` returns `MlResult<()>` — any failure in forward, loss, or backward propagates up and halts training
- `DataLoader::next_batch` returns `Option<(Tensor, Tensor)>` — epoch end is expressed as `None`, not an error
- Checkpoint load errors are reported as `MlError` — partial loads are rejected, not silently applied

### Performance
- `Scaler::fit` runs once on the training split; `transform` is a single pass — no per-batch normalization overhead
- `DataLoader` shuffles an index vec once per epoch — O(n) shuffle cost, zero copy of underlying data
- `Trainer` takes ownership of model and optimizer — avoids `Mutex`/`RwLock` overhead during the training loop

## Integration Points

| System | Integration | Notes |
|--------|-------------|-------|
| `mlautograd` | `tape::backward` drives the backward pass inside `Trainer::train`; `Tensor`, `MlError`, `MlResult` are re-exported | All gradient computation originates here |
| `mllayers` | `Layer` trait used for the model held by `Trainer`; activations and layer types re-exported | Layer parameters are collected via `Layer::parameters_mut()` for the optimizer |
| `mloptim` | `Optimizer::step` updates parameters after backward; schedulers and grad-clip utilities re-exported | Full mloptim surface re-exported through mltraining |
| `llmeco` architectures | Import `mltraining` as the single dep for the entire ml* stack | Architecture crates call `use mltraining::*` and never depend on mlautograd/mllayers/mloptim directly |

---

## See Also

- [Overview](../README.md)
- [Integration Guide](integration.md)
