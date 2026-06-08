// @covers: Loss
use mltraining::{Loss, MlResult, Tensor};

struct ZeroLoss;

impl Loss for ZeroLoss {
    fn forward(&self, _p: &Tensor, _t: &Tensor) -> MlResult<Tensor> {
        Ok(Tensor::zeros(vec![1]))
    }
}

#[test]
fn test_loss_forward_returns_scalar_tensor() {
    let loss = ZeroLoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("pred");
    let tgt = Tensor::zeros(vec![3]);
    let out = loss.forward(&pred, &tgt).expect("forward");
    assert_eq!(out.shape(), &[1]);
}

#[test]
fn test_loss_forward_zero_output_has_zero_value() {
    let loss = ZeroLoss;
    let pred = Tensor::ones(vec![2]);
    let tgt = Tensor::ones(vec![2]);
    let out = loss.forward(&pred, &tgt).expect("forward");
    assert!(out.to_vec()[0].abs() < 1e-6);
}
