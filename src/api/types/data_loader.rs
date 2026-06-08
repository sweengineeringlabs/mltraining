use crate::api::traits::dataset::Dataset;

/// An iterator that yields batched `(input, target)` tensor pairs.
pub struct DataLoader<D: Dataset> {
    pub(crate) dataset: D,
    pub(crate) batch_size: usize,
    pub(crate) shuffle: bool,
    pub(crate) indices: Vec<usize>,
    pub(crate) current_idx: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        Self { dataset, batch_size, shuffle, indices, current_idx: 0 }
    }

    pub fn reset(&mut self) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        self.current_idx = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    pub fn num_batches(&self) -> usize {
        let len = self.indices.len();
        if len == 0 { return 0; }
        (len + self.batch_size - 1) / self.batch_size
    }
}
