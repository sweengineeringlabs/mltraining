use mllayers::layer::Layer;

/// Produce a human-readable summary of a model's parameter budget.
pub fn model_summary(model: &dyn Layer) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use mllayers::{Linear, Sequential};

    #[test]
    fn test_summary_single_linear() {
        let layer = Linear::new(4, 3);
        let summary = model_summary(&layer);
        assert!(summary.contains("Total parameters:     15"));
        assert!(summary.contains("Trainable parameters: 15"));
        assert!(summary.contains("60 bytes"));
    }

    #[test]
    fn test_summary_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(10, 5)),
            Box::new(Linear::new(5, 2)),
        ]);
        let summary = model_summary(&model);
        assert!(summary.contains("Total parameters:     67"));
        assert!(summary.contains("Trainable parameters: 67"));
        assert!(summary.contains("268 bytes"));
    }
}
