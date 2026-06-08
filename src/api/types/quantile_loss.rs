/// Quantile Loss for quantile regression.
pub struct QuantileLoss {
    pub(crate) quantile: f32,
}

impl QuantileLoss {
    pub fn new(quantile: f32) -> Self {
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be in (0, 1), got {quantile}"
        );
        QuantileLoss { quantile }
    }
}

impl Default for QuantileLoss {
    fn default() -> Self { QuantileLoss::new(0.5) }
}
