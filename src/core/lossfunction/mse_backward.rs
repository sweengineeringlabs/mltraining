use mlautograd::{BackwardOp, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::loss::Loss;
use crate::api::types::m_s_e_loss::MSELoss;

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> MlResult<Tensor> {
        let diff = predictions.sub_raw(targets)?;
        let sq = diff.pow_raw(2.0);
        let mse_val = sq.mean_all_raw();
        let output = Tensor::from_vec(vec![mse_val], vec![1])
            .map_err(mlautograd::MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(MseBackward { n: predictions.numel() }),
                output_id: output.id(),
                input_ids: vec![predictions.id()],
                saved_tensors: vec![predictions.clone(), targets.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }
}

struct MseBackward { n: usize }

impl BackwardOp for MseBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];
        let diff = predictions.sub_raw(targets)
            .unwrap_or_else(|_| Tensor::zeros(predictions.shape().to_vec()));
        let scale = 2.0 / self.n as f32;
        let grad_pred = diff.mul_scalar_raw(scale);
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);
        vec![grad_pred]
    }

    fn name(&self) -> &str {
        let op_name = "MSEBackward";
        op_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: forward
    #[test]
    fn test_mse_loss_identical_inputs_returns_zero() {
        let loss = MSELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("pred");
        let tgt = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("tgt");
        let result = loss.forward(&pred, &tgt).expect("forward");
        assert!(result.to_vec()[0].abs() < 1e-6);
    }

    /// @covers: forward
    #[test]
    fn test_mse_loss_known_value() {
        let loss = MSELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("pred");
        let tgt = Tensor::from_vec(vec![3.0, 4.0], vec![2]).expect("tgt");
        let result = loss.forward(&pred, &tgt).expect("forward");
        assert!((result.to_vec()[0] - 4.0).abs() < 1e-6);
    }
}
