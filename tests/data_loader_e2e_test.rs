// @covers: DataLoader
use mltraining::{DataLoader, Dataset, MlResult, Tensor};

struct RangeDs { n: usize }

impl Dataset for RangeDs {
    fn len(&self) -> usize { self.n }
    fn get(&self, idx: usize) -> MlResult<(Tensor, Tensor)> {
        let i = Tensor::from_vec(vec![idx as f32; 2], vec![2]).expect("input");
        let t = Tensor::from_vec(vec![idx as f32], vec![1]).expect("target");
        Ok((i, t))
    }
    fn input_shape(&self) -> Vec<usize> { vec![2] }
    fn target_dim(&self) -> usize { 1 }
}

#[test]
fn test_data_loader_batch_count_ceiling_division() {
    let loader = DataLoader::new(RangeDs { n: 7 }, 3, false);
    assert_eq!(loader.num_batches(), 3);
}

#[test]
fn test_data_loader_iterates_all_samples() {
    let loader = DataLoader::new(RangeDs { n: 6 }, 2, false);
    let batches: Vec<_> = loader.collect();
    assert_eq!(batches.len(), 3);
    for (inputs, targets) in &batches {
        assert_eq!(inputs.shape()[0], 2);
        assert_eq!(targets.shape()[0], 2);
    }
}

#[test]
fn test_data_loader_reset_allows_second_iteration() {
    let mut loader = DataLoader::new(RangeDs { n: 4 }, 2, false);
    let first: Vec<_> = loader.by_ref().collect();
    assert!(loader.next().is_none());
    loader.reset();
    let second: Vec<_> = loader.collect();
    assert_eq!(first.len(), second.len());
}
