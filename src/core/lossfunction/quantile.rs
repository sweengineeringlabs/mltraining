use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::loss::Loss;
use crate::api::types::quantile_loss::QuantileLoss;

impl Loss for QuantileLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> MlResult<Tensor> {
        let diff = predictions.sub_raw(targets)?;
        let diff_data = diff.to_vec();
        let n = diff_data.len();
        let q = self.quantile;

        let loss_data: Vec<f32> = diff_data.iter().map(|&d| {
            let error = -d;
            if error >= 0.0 { q * error } else { (q - 1.0) * error }
        }).collect();

        let sum: f32 = loss_data.iter().sum();
        let mean = sum / n as f32;
        let output = Tensor::from_vec(vec![mean], vec![1])
            .map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(QuantileBackward { n, quantile: self.quantile }),
                output_id: output.id(),
                input_ids: vec![predictions.id()],
                saved_tensors: vec![predictions.clone(), targets.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }
}

struct QuantileBackward { n: usize, quantile: f32 }

impl BackwardOp for QuantileBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];
        let diff = predictions.sub_raw(targets)
            .unwrap_or_else(|_| Tensor::zeros(predictions.shape().to_vec()));
        let diff_data = diff.to_vec();
        let q = self.quantile;
        let n = self.n as f32;

        let grad_data: Vec<f32> = diff_data.iter().map(|&d| {
            let error = -d;
            if error >= 0.0 { -q / n } else { (1.0 - q) / n }
        }).collect();

        let grad_pred = Tensor::from_vec(grad_data, diff.shape().to_vec())
            .unwrap_or_else(|_| Tensor::zeros(diff.shape().to_vec()));
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);
        vec![grad_pred]
    }

    fn name(&self) -> &str {
        let op_name = "QuantileBackward";
        op_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: forward
    #[test]
    fn test_quantile_loss_identical_inputs_returns_zero() {
        let loss = QuantileLoss::new(0.5);
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("pred");
        let tgt = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("tgt");
        let result = loss.forward(&pred, &tgt).expect("forward");
        assert!(result.to_vec()[0].abs() < 1e-6);
    }

    /// @covers: forward
    #[test]
    fn test_quantile_loss_default_is_median() {
        let loss = QuantileLoss::default();
        let pred = Tensor::from_vec(vec![0.0], vec![1]).expect("pred");
        let tgt = Tensor::from_vec(vec![1.0], vec![1]).expect("tgt");
        let result = loss.forward(&pred, &tgt).expect("forward");
        assert!(result.to_vec()[0] > 0.0);
    }
}
