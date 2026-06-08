// @covers: MAELoss
use mltraining::{MAELoss, Loss, Tensor};

#[test]
fn test_mae_loss_re_exported_from_api() {
    let loss = MAELoss::new();
    let pred = Tensor::from_vec(vec![3.0, 1.0], vec![2]).expect("pred");
    let tgt = Tensor::from_vec(vec![1.0, 3.0], vec![2]).expect("tgt");
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 2.0).abs() < 1e-5);
}
