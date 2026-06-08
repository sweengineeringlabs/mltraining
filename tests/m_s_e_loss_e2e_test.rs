// @covers: MSELoss
use mltraining::{MSELoss, Loss, Tensor};

#[test]
fn test_mse_loss_re_exported_from_api() {
    let loss = MSELoss::new();
    let pred = Tensor::from_vec(vec![2.0, 4.0], vec![2]).expect("pred");
    let tgt = Tensor::from_vec(vec![0.0, 0.0], vec![2]).expect("tgt");
    // (4 + 16) / 2 = 10
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 10.0).abs() < 1e-5);
}
