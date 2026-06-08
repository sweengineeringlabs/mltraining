// @covers: mlautograd integration via mltraining
//
// Exercises mlautograd types (Tensor, tape) through mltraining's public API.
use mlautograd::{Tensor, tape};
use mltraining::{MSELoss, Loss};

#[test]
fn test_tensor_operations_via_mltraining() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor a");
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).expect("tensor b");
    let diff = a.sub_raw(&b).expect("sub");
    let data = diff.to_vec();
    assert_eq!(data, vec![-3.0, -3.0, -3.0]);
}

#[test]
fn test_tape_grad_recording_via_mltraining() {
    let pred = Tensor::from_vec(vec![0.0], vec![1]).expect("pred");
    let tgt = Tensor::from_vec(vec![1.0], vec![1]).expect("tgt");
    let loss_fn = MSELoss::new();

    tape::clear_tape();
    let loss = loss_fn.forward(&pred, &tgt).expect("forward");
    tape::backward(&loss);

    let grad = tape::grad(&pred);
    assert!(grad.is_some(), "gradient should exist after backward");
}

#[test]
fn test_no_grad_context_suppresses_recording() {
    let pred = Tensor::from_vec(vec![1.0], vec![1]).expect("pred");
    let tgt = Tensor::from_vec(vec![2.0], vec![1]).expect("tgt");
    let loss_fn = MSELoss::new();

    tape::clear_tape();
    tape::no_grad(|| {
        let _loss = loss_fn.forward(&pred, &tgt).expect("forward in no_grad");
    });

    let grad = tape::grad(&pred);
    assert!(grad.is_none(), "gradient should not exist inside no_grad");
}
