use anyhow::Result;
use ndarray::{Array3, Array4, Axis, Ix3};
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Session, SessionBuilder};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

pub struct Model {
    session: Session,
    pub(crate) vocab: Vec<String>,
}

impl Model {
    pub fn new_from_file<P: AsRef<Path>>(model_path: P, vocab_path: P) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        let vocab = load_vocab(BufReader::new(File::open(vocab_path)?))?;
        Ok(Self { session, vocab })
    }

    pub fn new_from_memory(model_data: &[u8], vocab_data: &[u8]) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(model_data)?;
        let vocab = load_vocab(BufReader::new(vocab_data))?;
        Ok(Self { session, vocab })
    }

    pub fn forward(&self, data: Array4<f32>) -> Result<Array3<f32>> {
        let outputs = self.session.run(inputs![data]?)?;
        let output_array = outputs
            .iter()
            .next()
            .unwrap()
            .1
            .try_extract_tensor::<f32>()?;
        Ok(output_array.softmax(Axis(2)).into_dimensionality::<Ix3>()?)
    }
}

fn load_vocab<R: Read>(vocab: BufReader<R>) -> Result<Vec<String>> {
    let mut label_mapping = vec!["".to_string(), "".to_string()];

    for line in vocab.lines() {
        label_mapping.push(line?);
    }
    Ok(label_mapping)
}
