use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::loss::Loss;

/// Cross-Entropy Loss (numerically stable log-softmax + NLL).
///
/// Predictions: raw logits [batch, classes].
/// Targets: one-hot encoded [batch, classes].
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self { Self }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self { Self::new() }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> MlResult<Tensor> {
        let pred_shape = predictions.shape().to_vec();
        let tgt_shape = targets.shape().to_vec();

        assert_eq!(pred_shape.len(), 2,
            "CrossEntropyLoss expects 2D predictions [batch, classes], got {:?}", pred_shape);
        assert_eq!(pred_shape, tgt_shape,
            "CrossEntropyLoss: predictions {:?} != targets {:?}", pred_shape, tgt_shape);

        let batch_size = pred_shape[0];
        let num_classes = pred_shape[1];

        let pred_data = predictions.to_vec();
        let tgt_data = targets.to_vec();
        let mut total_loss = 0.0f32;

        for b in 0..batch_size {
            let offset = b * num_classes;
            let row = &pred_data[offset..offset + num_classes];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;

            let mut sample_loss = 0.0f32;
            for c in 0..num_classes {
                let log_softmax = row[c] - log_sum_exp;
                sample_loss -= tgt_data[offset + c] * log_softmax;
            }
            total_loss += sample_loss;
        }

        let mean_loss = total_loss / batch_size as f32;
        let output = Tensor::from_vec(vec![mean_loss], vec![1])
            .map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(CrossEntropyBackward { batch_size, num_classes }),
                output_id: output.id(),
                input_ids: vec![predictions.id()],
                saved_tensors: vec![predictions.clone(), targets.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }
}

struct CrossEntropyBackward { batch_size: usize, num_classes: usize }

impl BackwardOp for CrossEntropyBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let pred_data = predictions.to_vec();
        let tgt_data = targets.to_vec();
        let batch_size = self.batch_size;
        let num_classes = self.num_classes;
        let mut grad_data = vec![0.0f32; batch_size * num_classes];

        for b in 0..batch_size {
            let offset = b * num_classes;
            let row = &pred_data[offset..offset + num_classes];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();

            for c in 0..num_classes {
                let softmax_val = exp_vals[c] / sum_exp;
                grad_data[offset + c] = (softmax_val - tgt_data[offset + c]) / batch_size as f32;
            }
        }

        let grad_pred = Tensor::from_vec(grad_data, predictions.shape().to_vec())
            .expect("cross_entropy backward grad tensor");
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);
        vec![grad_pred]
    }

    fn name(&self) -> &str { "CrossEntropyBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss_perfect_prediction_low_loss() {
        let loss = CrossEntropyLoss::new();
        let pred = Tensor::from_vec(vec![10.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let tgt = Tensor::from_vec(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        assert!(result.to_vec()[0] < 0.01);
    }

    #[test]
    fn test_cross_entropy_loss_wrong_prediction_high_loss() {
        let loss = CrossEntropyLoss::new();
        let pred = Tensor::from_vec(vec![0.0, 0.0, 10.0], vec![1, 3]).unwrap();
        let tgt = Tensor::from_vec(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        assert!(result.to_vec()[0] > 5.0);
    }
}
