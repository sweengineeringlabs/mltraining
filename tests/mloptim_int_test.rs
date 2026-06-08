// @covers: mloptim integration via mltraining
//
// Exercises mloptim types (Adam, AdamW, SGD) through mltraining's public API.
use mloptim::{Adam, AdamW, SGD};
use mltraining::{Trainer, MSELoss, Linear, TrainerOps, Tensor};

#[test]
fn test_adam_optimizer_via_trainer_train_epoch() {
    let mut trainer = Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).expect("input");
    let target = Tensor::from_vec(vec![1.0], vec![1, 1]).expect("target");
    let batches = vec![(input, target)];
    let loss = trainer.train_epoch(&batches).expect("train_epoch");
    assert!(loss.is_finite(), "loss should be finite: {loss}");
}

#[test]
fn test_adamw_optimizer_via_trainer() {
    let mut trainer = Trainer::new(Linear::new(2, 1), AdamW::new(0.01), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).expect("input");
    let target = Tensor::from_vec(vec![1.0], vec![1, 1]).expect("target");
    let batches = vec![(input, target)];
    let loss = trainer.train_epoch(&batches).expect("train_epoch");
    assert!(loss.is_finite(), "loss should be finite: {loss}");
}

#[test]
fn test_sgd_optimizer_via_trainer() {
    let mut trainer = Trainer::new(Linear::new(2, 1), SGD::new(0.01), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).expect("input");
    let target = Tensor::from_vec(vec![1.0], vec![1, 1]).expect("target");
    let batches = vec![(input, target)];
    let loss = trainer.train_epoch(&batches).expect("train_epoch");
    assert!(loss.is_finite(), "loss should be finite: {loss}");
}

#[test]
fn test_trainer_predict_returns_correct_shape() {
    let mut trainer = Trainer::new(Linear::new(3, 2), Adam::new(0.01), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("input");
    let output = trainer.predict(&input).expect("predict");
    assert_eq!(output.shape(), &[1, 2]);
}
