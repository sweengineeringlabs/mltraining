// @covers: HuberLoss
use mltraining::{HuberLoss, Loss, Tensor};

#[test]
fn test_huber_loss_identical_predictions_returns_zero() {
    let loss = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]).expect("pred");
    let tgt = pred.clone();
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!(result.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_huber_loss_large_error_uses_linear_regime() {
    let delta = 1.0_f32;
    let loss = HuberLoss::new(delta);
    // error = 5.0 >> delta → linear regime: delta*(|e| - 0.5*delta) = 1*(5 - 0.5) = 4.5
    let pred = Tensor::from_vec(vec![0.0], vec![1]).expect("pred");
    let tgt = Tensor::from_vec(vec![5.0], vec![1]).expect("tgt");
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 4.5).abs() < 1e-5, "got {}", result.to_vec()[0]);
}
