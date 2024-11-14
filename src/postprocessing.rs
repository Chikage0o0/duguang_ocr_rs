use anyhow::Result;
use ndarray::{s, Array2, ArrayView3, Axis};

use crate::model::Model;

impl Model {
    pub fn generate_output_strings(&self, preds: ArrayView3<f32>) -> Result<Vec<String>> {
        let label_mapping = &self.vocab;
        let preds = argmax_along_dim2(preds);
        Ok(preds_to_strings(preds, &label_mapping))
    }
}

fn argmax_along_dim2(input: ArrayView3<f32>) -> Array2<usize> {
    let dim1 = input.shape()[0];
    let dim2 = input.shape()[1];

    let mut result = Array2::<usize>::zeros((dim1, dim2));

    for i in 0..dim1 {
        for j in 0..dim2 {
            let slice = input.slice(s![i, j, ..]);
            let max_index = slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            result[(i, j)] = max_index;
        }
    }

    result
}

fn preds_to_strings(preds: Array2<usize>, label_mapping: &[String]) -> Vec<String> {
    let batch_size = preds.shape()[0];
    let mut final_str_list = Vec::new();

    for i in 0..batch_size {
        let pred_idx = preds.index_axis(Axis(0), i);
        let mut last_p = 0;
        let mut str_pred = Vec::new();
        for &p in pred_idx {
            if p != last_p && p != 0 {
                if let Some(label) = label_mapping.get(p) {
                    str_pred.push(label.clone());
                }
            }
            last_p = p;
        }
        let final_str = str_pred.join("");
        final_str_list.push(final_str);
    }

    final_str_list
}
