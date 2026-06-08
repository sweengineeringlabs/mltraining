use mllayers::layer::Layer;
use crate::api::types::model_summary::ModelSummary;

impl ModelSummary {
    /// Produce a human-readable summary of a model's parameter budget.
    pub(crate) fn summarize(model: &dyn Layer) -> String {
        let params = model.parameters();

        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        let trainable_params: usize = params.iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel())
            .sum();
        let memory_bytes = total_params * std::mem::size_of::<f32>();

        let mut out = String::new();
        out.push_str("===== Model Summary =====\n");
        out.push_str(&format!("Total parameters:     {total_params}\n"));
        out.push_str(&format!("Trainable parameters: {trainable_params}\n"));
        out.push_str(&format!("Memory estimate:      {} bytes", memory_bytes));

        if memory_bytes >= 1024 * 1024 {
            let mb = memory_bytes as f64 / (1024.0 * 1024.0);
            out.push_str(&format!(" ({mb:.2} MB)"));
        } else if memory_bytes >= 1024 {
            let kb = memory_bytes as f64 / 1024.0;
            out.push_str(&format!(" ({kb:.2} KB)"));
        }

        out.push('\n');
        out.push_str("=========================");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mllayers::Linear;

    /// @covers: summarize
    #[test]
    fn test_summarize_contains_total_parameters() {
        let layer = Linear::new(4, 3);
        let s = ModelSummary::summarize(&layer);
        assert!(s.contains("Total parameters:     15"), "got: {s}");
        assert!(s.contains("Trainable parameters: 15"));
        assert!(s.contains("60 bytes"));
    }
}
