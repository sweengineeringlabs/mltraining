/// Mean Absolute Error loss.
pub struct MAELoss;

impl MAELoss {
    pub fn new() -> Self { MAELoss }
}

impl Default for MAELoss {
    fn default() -> Self { MAELoss::new() }
}
