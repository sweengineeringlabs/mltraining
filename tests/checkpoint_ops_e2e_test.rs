// @covers: CheckpointOps::save_to, CheckpointOps::load_into_model
use mltraining::{CheckpointOps, Checkpoint, Linear, Layer};

#[test]
fn test_checkpoint_ops_save_to_and_load_into_model() {
    let src = Linear::new(2, 1);
    let ckpt = Checkpoint::from_model(&src, 3, 0.25);
    let tmp = std::env::temp_dir().join("test_checkpoint_ops.bin");

    ckpt.save_to(&tmp).expect("save_to");

    let loaded = Checkpoint::load(&tmp).expect("load");
    let mut dst = Linear::new(2, 1);
    loaded.load_into_model(&mut dst).expect("load_into_model");

    let src_data: Vec<f32> = src.parameters().iter().flat_map(|p| p.to_vec()).collect();
    let dst_data: Vec<f32> = dst.parameters().iter().flat_map(|p| p.to_vec()).collect();
    assert_eq!(src_data, dst_data, "parameters should match after restore");
    let _ = std::fs::remove_file(&tmp);
}
