/// Cross-Entropy Loss (numerically stable log-softmax + NLL).
///
/// Predictions: raw logits [batch, classes].
/// Targets: one-hot encoded [batch, classes].
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self { CrossEntropyLoss }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self { CrossEntropyLoss::new() }
}
