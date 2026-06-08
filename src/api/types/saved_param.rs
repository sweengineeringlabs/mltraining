/// A single serialized parameter tensor.
pub struct SavedParam {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
