// @covers: save_checkpoint, load_checkpoint, model_summary
use mltraining::{save_checkpoint, load_checkpoint, model_summary, Linear, Layer};

#[test]
fn test_save_and_load_checkpoint_round_trip() {
    let model = Linear::new(3, 1);
    let tmp = std::env::temp_dir().join("mltraining_svc_test_ckpt.bin");

    save_checkpoint(&model, &tmp, 1, 0.42).expect("save_checkpoint");
    let mut dst = Linear::new(3, 1);
    let ckpt = load_checkpoint(&mut dst, &tmp).expect("load_checkpoint");

    assert_eq!(ckpt.epoch, 1);
    assert!((ckpt.best_val_loss - 0.42).abs() < 1e-6);

    let src_params: Vec<f32> = model.parameters().iter().flat_map(|p| p.to_vec()).collect();
    let dst_params: Vec<f32> = dst.parameters().iter().flat_map(|p| p.to_vec()).collect();
    assert_eq!(src_params, dst_params, "restored params must match original");

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_model_summary_wraps_layer() {
    let layer = Linear::new(2, 2);
    let s = model_summary(&layer);
    // 2*2 weights + 2 biases = 6 params
    assert!(s.contains("Total parameters:     6"), "got: {s}");
}
