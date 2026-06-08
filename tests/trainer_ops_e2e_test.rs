// @covers: TrainerOps::train_epoch, TrainerOps::validate, TrainerOps::fit, TrainerOps::predict
use mltraining::{Trainer, MSELoss, Linear, Adam, TrainerOps, Tensor};

fn make_batch() -> Vec<(Tensor, Tensor)> {
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).expect("input");
    let target = Tensor::from_vec(vec![1.0], vec![1, 1]).expect("target");
    vec![(input, target)]
}

#[test]
fn test_trainer_ops_train_epoch_returns_finite_loss() {
    let mut trainer = Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
    let batches = make_batch();
    let loss = trainer.train_epoch(&batches).expect("train_epoch");
    assert!(loss.is_finite(), "loss should be finite: {loss}");
}

#[test]
fn test_trainer_ops_validate_returns_finite_loss() {
    let mut trainer = Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
    let batches = make_batch();
    let loss = trainer.validate(&batches).expect("validate");
    assert!(loss.is_finite(), "val_loss should be finite: {loss}");
}

#[test]
fn test_trainer_ops_fit_returns_history() {
    let mut trainer = Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
    let batches = make_batch();
    let history = trainer.fit(&batches, &batches, 3).expect("fit");
    assert_eq!(history.len(), 3, "should have 3 epochs in history");
}

#[test]
fn test_trainer_ops_predict_returns_correct_shape() {
    let mut trainer = Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).expect("input");
    let output = trainer.predict(&input).expect("predict");
    assert_eq!(output.shape(), &[1, 1]);
}
