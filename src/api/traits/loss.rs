use mlautograd::{MlResult, Tensor};

pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> MlResult<Tensor>;
}
