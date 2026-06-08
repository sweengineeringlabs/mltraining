// @covers: Dataset::is_empty
use mltraining::{Dataset, MlResult, Tensor};

struct EmptyDs;

impl Dataset for EmptyDs {
    fn len(&self) -> usize { 0 }
    fn get(&self, _: usize) -> MlResult<(Tensor, Tensor)> { unimplemented!() }
    fn input_shape(&self) -> Vec<usize> { vec![1] }
    fn target_dim(&self) -> usize { 1 }
}

struct NonEmptyDs;

impl Dataset for NonEmptyDs {
    fn len(&self) -> usize { 3 }
    fn get(&self, idx: usize) -> MlResult<(Tensor, Tensor)> {
        let i = Tensor::from_vec(vec![idx as f32], vec![1]).expect("input");
        let t = Tensor::from_vec(vec![idx as f32], vec![1]).expect("target");
        Ok((i, t))
    }
    fn input_shape(&self) -> Vec<usize> { vec![1] }
    fn target_dim(&self) -> usize { 1 }
}

#[test]
fn test_is_empty_default_delegates_to_len() {
    assert!(EmptyDs.is_empty());
}

#[test]
fn test_is_empty_returns_false_for_non_empty_dataset() {
    assert!(!NonEmptyDs.is_empty());
}
