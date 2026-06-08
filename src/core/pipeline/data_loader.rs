use mlautograd::Tensor;
use crate::api::traits::dataset::Dataset;
use crate::api::types::data_loader::DataLoader;

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end];
        self.current_idx = end;

        let actual_batch_size = batch_indices.len();
        let input_shape = self.dataset.input_shape();
        let target_dim = self.dataset.target_dim();
        let input_elements: usize = input_shape.iter().product();

        let mut input_data = Vec::with_capacity(actual_batch_size * input_elements);
        let mut target_data = Vec::with_capacity(actual_batch_size * target_dim);

        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((input, target)) => {
                    input_data.extend_from_slice(&input.to_vec());
                    target_data.extend_from_slice(&target.to_vec());
                }
                Err(_) => continue,
            }
        }

        let mut batch_input_shape = vec![actual_batch_size];
        batch_input_shape.extend_from_slice(&input_shape);

        let input = Tensor::from_vec(input_data, batch_input_shape).ok()?;
        let target = Tensor::from_vec(target_data, vec![actual_batch_size, target_dim]).ok()?;

        Some((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlautograd::{MlResult, Tensor};
    use crate::api::traits::dataset::Dataset;

    struct MockDataLoader { n: usize }

    impl Dataset for MockDataLoader {
        fn len(&self) -> usize { self.n }
        fn get(&self, idx: usize) -> MlResult<(Tensor, Tensor)> {
            let input = Tensor::from_vec(vec![idx as f32; 3], vec![3])
                .expect("mock input");
            let target = Tensor::from_vec(vec![idx as f32], vec![1])
                .expect("mock target");
            Ok((input, target))
        }
        fn input_shape(&self) -> Vec<usize> { vec![3] }
        fn target_dim(&self) -> usize { 1 }
    }

    /// @covers: next
    #[test]
    fn test_dataloader_iterates_all_samples() {
        let ds = MockDataLoader { n: 10 };
        let loader = DataLoader::new(ds, 4, false);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 3);
    }

    /// @covers: next
    #[test]
    fn test_next_returns_none_when_exhausted() {
        let ds = MockDataLoader { n: 2 };
        let mut loader = DataLoader::new(ds, 4, false);
        assert!(loader.next().is_some());
        assert!(loader.next().is_none());
    }
}
