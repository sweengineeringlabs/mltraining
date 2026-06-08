// @covers: Trainer::new, Trainer::with_grad_clip, Trainer::with_early_stopping,
//          Trainer::with_scheduler, Trainer::with_checkpoint_dir
use mltraining::{Trainer, MSELoss, Linear, Adam, TrainerOps, Tensor};

fn make_trainer() -> Trainer<Linear, Adam, MSELoss> {
    Trainer::new(Linear::new(2, 1), Adam::new(0.01), MSELoss::new())
}

#[test]
fn test_trainer_new_default_options_are_none() {
    let t = make_trainer();
    assert!(t.grad_clip_norm.is_none());
    assert!(t.patience.is_none());
    assert!(t.scheduler.is_none());
    assert!(t.checkpoint_dir.is_none());
}

#[test]
fn test_trainer_with_grad_clip_sets_value() {
    let t = make_trainer().with_grad_clip(0.5);
    assert_eq!(t.grad_clip_norm, Some(0.5));
}

#[test]
fn test_trainer_with_early_stopping_sets_patience() {
    let t = make_trainer().with_early_stopping(10);
    assert_eq!(t.patience, Some(10));
}

#[test]
fn test_trainer_with_checkpoint_dir_sets_path() {
    let t = make_trainer().with_checkpoint_dir("/tmp/test_ckpt");
    assert_eq!(t.checkpoint_dir.as_deref(), Some("/tmp/test_ckpt"));
}

#[test]
fn test_trainer_train_epoch_reduces_loss() {
    let mut trainer = Trainer::new(Linear::new(1, 1), Adam::new(0.1), MSELoss::new());
    let input = Tensor::from_vec(vec![1.0], vec![1, 1]).expect("input");
    let target = Tensor::from_vec(vec![2.0], vec![1, 1]).expect("target");
    let batches = vec![(input.clone(), target.clone())];

    let loss1 = trainer.train_epoch(&batches).expect("epoch 1");
    let loss2 = trainer.train_epoch(&batches).expect("epoch 2");

    // After two steps the model should be closer to the target
    assert!(loss2 <= loss1 + 1e-4, "loss should not increase: {loss1} -> {loss2}");
}
