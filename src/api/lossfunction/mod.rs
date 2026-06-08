#![allow(unused_imports)]

pub mod cross_entropy;
pub mod cross_entropy_loss;
pub mod huber;
pub mod huber_loss;
pub mod mae_backward;
pub mod m_a_e_loss;
pub mod mse_backward;
pub mod m_s_e_loss;
pub mod quantile;
pub mod quantile_loss;

pub use crate::api::traits::loss::Loss;
pub use crate::api::types::cross_entropy_loss::CrossEntropyLoss;
pub use crate::api::types::huber_loss::HuberLoss;
pub use crate::api::types::m_a_e_loss::MAELoss;
pub use crate::api::types::m_s_e_loss::MSELoss;
pub use crate::api::types::quantile_loss::QuantileLoss;
