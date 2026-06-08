/// Huber Loss with configurable delta.
pub struct HuberLoss {
    pub(crate) delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self { HuberLoss { delta } }
}

impl Default for HuberLoss {
    fn default() -> Self { HuberLoss::new(1.0) }
}
