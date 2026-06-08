// @covers: ModelSummary
use mltraining::{model_summary, Linear};

#[test]
fn test_model_summary_contains_total_parameters() {
    // Linear(3, 2): 3*2 weights + 2 biases = 8
    let layer = Linear::new(3, 2);
    let s = model_summary(&layer);
    assert!(s.contains("Total parameters:     8"), "got: {s}");
}

#[test]
fn test_model_summary_contains_trainable_parameters() {
    let layer = Linear::new(4, 1);
    let s = model_summary(&layer);
    // 4*1 + 1 = 5 params
    assert!(s.contains("Trainable parameters: 5"), "got: {s}");
}

#[test]
fn test_model_summary_contains_memory_estimate() {
    let layer = Linear::new(2, 2);
    let s = model_summary(&layer);
    assert!(s.contains("Memory estimate:"), "got: {s}");
}
