// @covers: QuantileLoss
use mltraining::{QuantileLoss, Loss, Tensor};

#[test]
fn test_quantile_loss_identical_predictions_returns_zero() {
    let loss = QuantileLoss::new(0.5);
    let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("pred");
    let tgt = pred.clone();
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!(result.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_quantile_loss_returns_finite_scalar() {
    let loss = QuantileLoss::new(0.75);
    let pred = Tensor::from_vec(vec![1.0, 3.0, 2.0], vec![3]).expect("pred");
    let tgt = Tensor::from_vec(vec![2.0, 2.0, 4.0], vec![3]).expect("tgt");
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!(result.to_vec()[0].is_finite());
    assert_eq!(result.shape(), &[1]);
}
