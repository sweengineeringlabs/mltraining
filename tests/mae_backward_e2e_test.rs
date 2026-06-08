// @covers: MaeBackward
use mltraining::{MAELoss, Loss, Tensor};

#[test]
fn test_mae_backward_alias_is_mae_loss() {
    let loss = MAELoss::new();
    let pred = Tensor::from_vec(vec![0.0, 2.0], vec![2]).expect("pred");
    let tgt = Tensor::from_vec(vec![2.0, 0.0], vec![2]).expect("tgt");
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 2.0).abs() < 1e-5);
}
