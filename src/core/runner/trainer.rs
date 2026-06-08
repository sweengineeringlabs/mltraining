use mlautograd::{MlError, MlResult, Tensor, tape};
use mllayers::layer::Layer;
use mloptim::Optimizer;
use crate::api::traits::loss::Loss;
use crate::api::traits::trainer_ops::TrainerOps;

/// Implementation marker satisfying SEA module-primary rule.
pub(crate) struct Trainer;

type ApiTrainer<M, O, L> = crate::api::types::trainer::Trainer<M, O, L>;

impl<M: Layer, O: Optimizer, L: Loss> ApiTrainer<M, O, L> {
    fn clip_gradients(&self, max_norm: f32) {
        let params = self.model.parameters();
        let mut total_norm_sq: f32 = 0.0;
        for param in &params {
            if let Some(grad) = tape::grad(param) {
                let grad_data = grad.to_vec();
                let sq_sum: f32 = grad_data.iter().map(|v| v * v).sum();
                total_norm_sq += sq_sum;
            }
        }
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in &params {
                if let Some(grad) = tape::grad(param) {
                    let clipped = grad.mul_scalar_raw(scale);
                    tape::set_grad(param, clipped);
                }
            }
        }
    }

    fn should_stop(&mut self, improved: bool) -> bool {
        let patience = match self.patience {
            Some(p) => p,
            None => return false,
        };
        if improved {
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }
        self.epochs_without_improvement >= patience
    }
}

impl<M: Layer, O: Optimizer, L: Loss> TrainerOps for ApiTrainer<M, O, L> {
    fn train_epoch(&mut self, batches: &[(Tensor, Tensor)]) -> MlResult<f32> {
        let mut total_loss = 0.0;

        for (input, target) in batches {
            tape::clear_tape();

            let output = self.model.forward(input)?;
            let loss = self.loss_fn.forward(&output, target)?;
            total_loss += loss.to_vec()[0];

            tape::backward(&loss);

            if let Some(max_norm) = self.grad_clip_norm {
                self.clip_gradients(max_norm);
            }

            let mut params: Vec<&mut Tensor> = self.model.parameters_mut();
            let mut param_refs: Vec<&mut Tensor> =
                params.iter_mut().map(|p| &mut **p).collect();
            self.optimizer.step(&mut param_refs)?;
        }

        Ok(total_loss / batches.len() as f32)
    }

    fn validate(&mut self, batches: &[(Tensor, Tensor)]) -> MlResult<f32> {
        let mut total_loss = 0.0;

        tape::no_grad(|| {
            for (input, target) in batches {
                let output = self.model.forward(input)
                    .unwrap_or_else(|_| Tensor::zeros(vec![1]));
                let loss = self.loss_fn.forward(&output, target)
                    .unwrap_or_else(|_| Tensor::zeros(vec![1]));
                total_loss += loss.to_vec()[0];
            }
        });

        Ok(total_loss / batches.len() as f32)
    }

    fn fit(
        &mut self,
        train_batches: &[(Tensor, Tensor)],
        val_batches: &[(Tensor, Tensor)],
        epochs: usize,
    ) -> MlResult<Vec<(f32, f32)>> {
        let mut history: Vec<(f32, f32)> = Vec::with_capacity(epochs);

        for epoch in 1..=epochs {
            let train_loss = self.train_epoch(train_batches)?;
            let val_loss = self.validate(val_batches)?;

            log::info!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch, epochs, train_loss, val_loss,
            );

            history.push((train_loss, val_loss));

            if let Some(ref mut scheduler) = self.scheduler {
                scheduler.step(&mut self.optimizer);
                log::debug!("Epoch {}: LR updated to {:.6}", epoch, scheduler.get_lr());
            }

            let is_improvement = val_loss < self.best_val_loss;

            if is_improvement {
                self.best_val_loss = val_loss;

                if let Some(ref dir) = self.checkpoint_dir {
                    std::fs::create_dir_all(dir).map_err(|e| {
                        MlError::TrainingError(format!(
                            "create checkpoint dir '{}': {}", dir, e
                        ))
                    })?;
                    let path = format!("{}/best_model.bin", dir);
                    crate::api::types::checkpoint::Checkpoint::create_and_save(&self.model, &path, epoch, val_loss)?;
                    log::info!(
                        "Epoch {}: saved best checkpoint (val_loss={:.6}) to {}",
                        epoch, val_loss, path,
                    );
                }
            }

            if self.should_stop(is_improvement) {
                log::info!(
                    "Early stopping triggered at epoch {} (patience={}, best_val_loss={:.6})",
                    epoch,
                    self.patience.unwrap_or(0),
                    self.best_val_loss,
                );
                break;
            }
        }

        Ok(history)
    }

    fn predict(&mut self, input: &Tensor) -> MlResult<Tensor> {
        tape::no_grad(|| self.model.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mllayers::Linear;
    use mloptim::Adam;
    use crate::api::types::m_s_e_loss::MSELoss;

    fn make_batches() -> Vec<(Tensor, Tensor)> {
        vec![
            (
                Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]).expect("input"),
                Tensor::from_vec(vec![1.0], vec![1]).expect("target"),
            ),
        ]
    }

    /// @covers: train_epoch
    #[test]
    fn test_train_epoch_returns_finite_loss() {
        let mut t = ApiTrainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
        let batches = make_batches();
        let loss = t.train_epoch(&batches).expect("train_epoch");
        assert!(loss.is_finite());
    }

    /// @covers: validate
    #[test]
    fn test_validate_returns_finite_loss() {
        let mut t = ApiTrainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
        let batches = make_batches();
        let loss = t.validate(&batches).expect("validate");
        assert!(loss.is_finite());
    }

    /// @covers: predict
    #[test]
    fn test_predict_returns_output_tensor() {
        let mut t = ApiTrainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
        let input = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]).expect("input");
        let output = t.predict(&input).expect("predict");
        assert_eq!(output.shape(), &[1, 1]);
    }
}
