// @covers: Loss::forward
use mltraining::{Loss, Tensor};

struct ZeroLoss;

impl Loss for ZeroLoss {
    fn forward(&self, _predictions: &Tensor, _targets: &Tensor) -> mltraining::MlResult<Tensor> {
        Ok(Tensor::zeros(vec![1]))
    }
}

#[test]
fn test_loss_trait_forward_returns_scalar() {
    let loss = ZeroLoss;
    let pred = Tensor::ones(vec![4]);
    let tgt = Tensor::zeros(vec![4]);
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert_eq!(result.shape(), &[1]);
}
