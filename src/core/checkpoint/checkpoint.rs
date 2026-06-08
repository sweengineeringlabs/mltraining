use mlautograd::MlResult;
use mllayers::layer::Layer;
use std::path::Path;
use crate::api::types::checkpoint::Checkpoint;
use crate::api::traits::checkpoint_ops::CheckpointOps;

impl CheckpointOps for Checkpoint {
    fn save_to<P: AsRef<Path>>(&self, path: P) -> MlResult<()> {
        self.save(path)
    }

    fn load_into_model(&self, model: &mut dyn Layer) -> MlResult<()> {
        Checkpoint::load_into_model(self, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mllayers::Linear;

    /// @covers: save_to
    #[test]
    fn test_save_to_via_checkpoint_ops_trait() {
        let layer = Linear::new(2, 1);
        let ckpt = Checkpoint::from_model(&layer, 3, 0.9);
        let tmp = std::env::temp_dir().join("test_checkpoint_ops_save_to.bin");
        ckpt.save_to(&tmp).expect("save_to failed");
        assert!(tmp.exists());
        let _ = std::fs::remove_file(&tmp);
    }

    /// @covers: load_into_model
    #[test]
    fn test_load_into_model_via_checkpoint_ops_trait() {
        let src = Linear::new(2, 1);
        let ckpt = Checkpoint::from_model(&src, 0, 0.0);
        let tmp = std::env::temp_dir().join("test_checkpoint_ops_load_into.bin");
        ckpt.save_to(&tmp).expect("save_to");
        let mut dst = Linear::new(2, 1);
        let loaded = Checkpoint::load(&tmp).expect("load");
        loaded.load_into_model(&mut dst).expect("load_into_model");
        let src_data: Vec<f32> = src.parameters().iter().flat_map(|p| p.to_vec()).collect();
        let dst_data: Vec<f32> = dst.parameters().iter().flat_map(|p| p.to_vec()).collect();
        assert_eq!(src_data, dst_data);
        let _ = std::fs::remove_file(&tmp);
    }
}
