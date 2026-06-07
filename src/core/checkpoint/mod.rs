use mlautograd::{MlError, MlResult, Tensor};
use mllayers::layer::Layer;
use std::fs;
use std::io::{Read, Cursor};
use std::path::Path;

pub struct Checkpoint {
    pub param_data: Vec<SavedParam>,
    pub epoch: usize,
    pub best_val_loss: f32,
}

pub struct SavedParam {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Checkpoint {
    pub fn from_model(model: &dyn Layer, epoch: usize, best_val_loss: f32) -> Self {
        let params = model.parameters();
        let param_data = params.iter().map(|p| SavedParam {
            data: p.to_vec(),
            shape: p.shape().to_vec(),
        }).collect();
        Checkpoint { param_data, epoch, best_val_loss }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> MlResult<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.epoch as u64).to_le_bytes());
        buf.extend_from_slice(&self.best_val_loss.to_le_bytes());
        buf.extend_from_slice(&(self.param_data.len() as u64).to_le_bytes());
        for param in &self.param_data {
            buf.extend_from_slice(&(param.shape.len() as u64).to_le_bytes());
            for &dim in &param.shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(param.data.len() as u64).to_le_bytes());
            for &val in &param.data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        fs::write(path, &buf)
            .map_err(|e| MlError::TrainingError(format!("save checkpoint: {e}")))?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> MlResult<Self> {
        let data = fs::read(path)
            .map_err(|e| MlError::TrainingError(format!("load checkpoint: {e}")))?;
        let mut cursor = Cursor::new(&data);

        let read_u64 = |cursor: &mut Cursor<&Vec<u8>>| -> MlResult<u64> {
            let mut buf = [0u8; 8];
            cursor.read_exact(&mut buf)
                .map_err(|e| MlError::TrainingError(format!("read u64: {e}")))?;
            Ok(u64::from_le_bytes(buf))
        };
        let read_f32_val = |cursor: &mut Cursor<&Vec<u8>>| -> MlResult<f32> {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf)
                .map_err(|e| MlError::TrainingError(format!("read f32: {e}")))?;
            Ok(f32::from_le_bytes(buf))
        };

        let epoch = read_u64(&mut cursor)? as usize;
        let best_val_loss = read_f32_val(&mut cursor)?;
        let num_params = read_u64(&mut cursor)? as usize;

        let mut param_data = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            let num_dims = read_u64(&mut cursor)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut cursor)? as usize);
            }
            let num_floats = read_u64(&mut cursor)? as usize;
            let mut floats = Vec::with_capacity(num_floats);
            for _ in 0..num_floats {
                floats.push(read_f32_val(&mut cursor)?);
            }
            param_data.push(SavedParam { data: floats, shape });
        }

        Ok(Checkpoint { param_data, epoch, best_val_loss })
    }

    pub fn load_into_model(&self, model: &mut dyn Layer) -> MlResult<()> {
        let mut params = model.parameters_mut();
        if params.len() != self.param_data.len() {
            return Err(MlError::InvalidConfig(format!(
                "checkpoint has {} params, model has {}",
                self.param_data.len(), params.len()
            )));
        }
        for (param, saved) in params.iter_mut().zip(self.param_data.iter()) {
            let tensor = Tensor::from_vec(saved.data.clone(), saved.shape.clone())
                .map_err(MlError::TensorError)?;
            param.update_data_from(&tensor);
        }
        Ok(())
    }
}

pub fn save_checkpoint<P: AsRef<Path>>(
    model: &dyn Layer,
    path: P,
    epoch: usize,
    best_val_loss: f32,
) -> MlResult<()> {
    let checkpoint = Checkpoint::from_model(model, epoch, best_val_loss);
    checkpoint.save(path)
}

pub fn load_checkpoint<P: AsRef<Path>>(
    model: &mut dyn Layer,
    path: P,
) -> MlResult<Checkpoint> {
    let checkpoint = Checkpoint::load(path)?;
    checkpoint.load_into_model(model)?;
    Ok(checkpoint)
}
