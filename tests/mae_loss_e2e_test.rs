// @covers: MAELoss
use mltraining::{MAELoss, Loss, Tensor};

#[test]
fn test_mae_loss_identical_predictions_returns_zero() {
    let loss = MAELoss::new();
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("pred");
    let tgt = pred.clone();
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!(result.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_mae_loss_known_value() {
    let loss = MAELoss::new();
    let pred = Tensor::from_vec(vec![1.0, 3.0], vec![2]).expect("pred");
    let tgt = Tensor::from_vec(vec![3.0, 1.0], vec![2]).expect("tgt");
    // |1-3| + |3-1| = 4, / 2 = 2.0
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 2.0).abs() < 1e-6, "got {}", result.to_vec()[0]);
}
