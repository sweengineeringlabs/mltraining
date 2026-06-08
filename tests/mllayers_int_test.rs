// @covers: mllayers integration via mltraining
//
// Exercises mllayers types through mltraining's public API.
use mllayers::{Linear, Sequential, Layer};
use mltraining::{Tensor, model_summary};

#[test]
fn test_linear_forward_produces_correct_output_shape() {
    let mut layer = Linear::new(4, 2);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).expect("input");
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_sequential_forward_chains_layers() {
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Linear::new(3, 1)),
    ]);
    let input = Tensor::from_vec(vec![1.0, 0.5, -1.0, 2.0], vec![1, 4]).expect("input");
    let output = model.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[1, 1]);
}

#[test]
fn test_model_summary_via_mllayers_layer() {
    let model = Linear::new(3, 2);
    let summary = model_summary(&model);
    // 3*2 weights + 2 biases = 8 params
    assert!(summary.contains("Total parameters:     8"));
    assert!(summary.contains("Trainable parameters: 8"));
}

#[test]
fn test_layer_parameters_count_matches_architecture() {
    let layer = Linear::new(5, 3);
    // 5*3 weights + 3 biases = 18
    let total: usize = layer.parameters().iter().map(|p| p.numel()).sum();
    assert_eq!(total, 18);
}
