// @covers: MseBackward
use mltraining::{MSELoss, Loss, Tensor};

#[test]
fn test_mse_backward_alias_is_mse_loss() {
    let loss = MSELoss::new();
    let pred = Tensor::from_vec(vec![0.0, 0.0], vec![2]).expect("pred");
    let tgt = Tensor::from_vec(vec![2.0, 4.0], vec![2]).expect("tgt");
    let result = loss.forward(&pred, &tgt).expect("forward");
    assert!((result.to_vec()[0] - 10.0).abs() < 1e-5);
}
