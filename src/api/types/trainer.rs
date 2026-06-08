use mloptim::LRScheduler;

/// The main training orchestrator.
pub struct Trainer<M, O, L> {
    pub(crate) model: M,
    pub(crate) optimizer: O,
    pub(crate) loss_fn: L,
    pub grad_clip_norm: Option<f32>,
    pub patience: Option<usize>,
    pub(crate) best_val_loss: f32,
    pub(crate) epochs_without_improvement: usize,
    pub scheduler: Option<Box<dyn LRScheduler>>,
    pub checkpoint_dir: Option<String>,
}

impl<M, O, L> Trainer<M, O, L> {
    pub fn new(model: M, optimizer: O, loss_fn: L) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            grad_clip_norm: None,
            patience: None,
            best_val_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            scheduler: None,
            checkpoint_dir: None,
        }
    }

    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.grad_clip_norm = Some(max_norm);
        self
    }

    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }

    pub fn with_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn with_checkpoint_dir(mut self, path: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(path.into());
        self
    }
}
