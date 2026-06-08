/// Mean Squared Error loss.
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self { MSELoss }
}

impl Default for MSELoss {
    fn default() -> Self { MSELoss::new() }
}
