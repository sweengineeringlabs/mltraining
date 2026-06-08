/// The kind of feature scaling algorithm to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalerType {
    MinMax,
    Standard,
    Robust,
}
