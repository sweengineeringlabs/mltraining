// @covers: CrossEntropyLoss
use mltraining::{CrossEntropyLoss, Loss, Tensor};

#[test]
fn test_cross_entropy_loss_perfect_prediction_is_low() {
    let loss = CrossEntropyLoss::new();
    let logits = Tensor::from_vec(vec![10.0, -10.0], vec![1, 2]).expect("logits");
    let targets = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]).expect("targets");
    let result = loss.forward(&logits, &targets).expect("forward");
    assert!(result.to_vec()[0] < 0.01, "expected low loss, got {}", result.to_vec()[0]);
}

#[test]
fn test_cross_entropy_loss_returns_finite_value() {
    let loss = CrossEntropyLoss::new();
    let logits = Tensor::from_vec(vec![0.5, 0.3, 0.2], vec![1, 3]).expect("logits");
    let targets = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![1, 3]).expect("targets");
    let result = loss.forward(&logits, &targets).expect("forward");
    assert!(result.to_vec()[0].is_finite());
}
