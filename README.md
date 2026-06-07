# mltraining

> **TLDR:** End-to-end training orchestration — loss functions, data loading, trainer loop, checkpointing, and re-exports of the full ml* stack. See [Overview](docs/README.md) for details.

## Table of Contents
- [Quick Start](#quick-start)
- [API](#api)
- [Documentation](#documentation)

## Quick Start

```rust
use mltraining::{Trainer, MSELoss, DataLoader};
use mloptim::Adam;

let loader = DataLoader::new(&dataset, 32, true);
let loss_fn = MSELoss::new();
let optimizer = Adam::new(&mut model.parameters_mut(), 1e-3);
let mut trainer = Trainer::new(model, loss_fn, optimizer, loader);
trainer.train(10)?;
```

## API

| Type | Description |
|------|-------------|
| `Loss` | Implement to define a differentiable loss function |
| `Dataset` | Implement to define a data source for the trainer |
| `MSELoss` | Mean squared error, standard regression loss |
| `HuberLoss` | Robust regression loss, less sensitive to outliers |
| `CrossEntropyLoss` | Classification loss with softmax normalization |
| `DataLoader` | Batches a `Dataset` with optional shuffling |
| `Trainer` | Orchestrates forward, loss, backward, optimizer step, metrics |
| `Checkpoint` | Serialize/deserialize model parameters to disk |
| `Scaler` | Normalize input features (MinMax or StandardScaler) |

## Documentation

- [Overview](docs/README.md) - W³H
- [Architecture](docs/3-design/architecture.md) - System design
- [Integration](docs/3-design/integration.md) - Integration guide

## Related FRs

- None (foundational crate)
