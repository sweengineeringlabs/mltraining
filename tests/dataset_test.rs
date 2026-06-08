// @covers: Dataset
use mltraining::{Dataset, MlResult, Tensor};

struct SingleRowDs {
    value: f32,
}

impl Dataset for SingleRowDs {
    fn len(&self) -> usize { 1 }
    fn get(&self, _: usize) -> MlResult<(Tensor, Tensor)> {
        let i = Tensor::from_vec(vec![self.value], vec![1]).expect("input");
        let t = Tensor::from_vec(vec![self.value * 2.0], vec![1]).expect("target");
        Ok((i, t))
    }
    fn input_shape(&self) -> Vec<usize> { vec![1] }
    fn target_dim(&self) -> usize { 1 }
}

#[test]
fn test_dataset_len_returns_correct_count() {
    let ds = SingleRowDs { value: 1.0 };
    assert_eq!(ds.len(), 1);
}

#[test]
fn test_dataset_get_returns_correct_shapes() {
    let ds = SingleRowDs { value: 3.0 };
    let (input, target) = ds.get(0).expect("get");
    assert_eq!(input.shape(), &[1]);
    assert_eq!(target.shape(), &[1]);
}

#[test]
fn test_dataset_is_empty_on_empty_len() {
    struct EmptyDs;
    impl Dataset for EmptyDs {
        fn len(&self) -> usize { 0 }
        fn get(&self, _: usize) -> MlResult<(Tensor, Tensor)> { unimplemented!() }
        fn input_shape(&self) -> Vec<usize> { vec![1] }
        fn target_dim(&self) -> usize { 1 }
    }
    assert!(EmptyDs.is_empty());
    assert!(!SingleRowDs { value: 0.0 }.is_empty());
}
