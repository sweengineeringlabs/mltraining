// @covers: model_summary, ModelSummary::summarize
use mltraining::{model_summary, ModelSummary, Linear, Sequential};

#[test]
fn test_free_fn_model_summary_single_linear() {
    let layer = Linear::new(4, 3);
    let summary = model_summary(&layer);
    assert!(summary.contains("Total parameters:     15"));
    assert!(summary.contains("Trainable parameters: 15"));
    assert!(summary.contains("60 bytes"));
}

#[test]
fn test_model_summary_summarize_sequential() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(10, 5)),
        Box::new(Linear::new(5, 2)),
    ]);
    let summary = model_summary(&model);
    assert!(summary.contains("Total parameters:     67"));
    assert!(summary.contains("Trainable parameters: 67"));
    assert!(summary.contains("268 bytes"));
}
