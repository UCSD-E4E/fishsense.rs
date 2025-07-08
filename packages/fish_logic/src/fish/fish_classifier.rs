use anyhow::Context;
use app_dirs2::{AppDataType, AppInfo, app_root};
use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_npy::read_npy;
use ort::Session;
use reqwest::blocking::get;
use serde::Deserialize;
use serde::de::{self, Deserializer, Error as DeError, Visitor};
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, create_dir_all};
use std::io::copy;
use std::path::PathBuf;

const MODEL_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/fish_classifier.onnx?download=true";
const EMBEDDING_URL: &str =
    "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/embeddings.npy?download=true";
const METADATA_URL: &str =
    "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/database.json?download=true";

#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub internal_ids: Vec<usize>,

    #[serde(default)]
    #[allow(dead_code)]
    pub image_ids: Vec<String>,

    #[serde(default)]
    #[allow(dead_code)]
    pub annotation_ids: Vec<String>,

    #[serde(default)]
    #[allow(dead_code)]
    pub drawn_fish_ids: Vec<u32>,

    #[serde(deserialize_with = "deserialize_keys_int_to_string")]
    pub keys: HashMap<String, serde_json::Value>,
}

pub struct FishClassifier {
    model: Session,
    embeddings: Array2<f32>,
    metadata: Metadata,
}

impl FishClassifier {
    pub fn new() -> anyhow::Result<Self> {
        let model_path = Self::download_file("fish_classifier.onnx", MODEL_URL)?;
        let embed_path = Self::download_file("embeddings.npy", EMBEDDING_URL)?;
        let meta_path = Self::download_file("database.json", METADATA_URL)?;

        // Debug: Print first few lines of the JSON file
        let metadata_content = std::fs::read_to_string(&meta_path)?;
        println!("First 5 lines of metadata file:");
        for line in metadata_content.lines().take(5) {
            println!("{}", line);
        }

        let model = Session::builder()?.commit_from_file(&model_path)?;

        println!("Model inputs:");
        for inp in &model.inputs {
            println!("- Name: {}, Type: {:?}", inp.name, inp.input_type);
        }

        let embeddings: Array2<f32> = read_npy(embed_path)?;
        let metadata_file = File::open(&meta_path)?;
        let metadata: Metadata = serde_json::from_reader(metadata_file)
            .context(format!("Failed to parse metadata file at {:?}", meta_path))?;

        Ok(FishClassifier {
            model,
            embeddings,
            metadata,
        })
    }

    fn download_file(filename: &str, url: &str) -> anyhow::Result<PathBuf> {
        let mut path = app_root(
            AppDataType::UserCache,
            &AppInfo {
                name: "fish-classifier",
                author: "unushri",
            },
        )?;
        path.push("models");
        create_dir_all(&path)?;
        path.push(filename);

        if !path.exists() {
            println!("Downloading {}...", filename);
            let mut resp = get(url)?;
            let mut out = File::create(&path)?;
            copy(&mut resp, &mut out)?;
        }

        Ok(path)
    }

    pub fn classify(&self, input_tensor: Array3<f32>) -> anyhow::Result<Vec<(String, f32)>> {
        let batched = input_tensor.insert_axis(Axis(0));

        let input_name = self.model.inputs[0].name.clone();
        println!("Using input name: {}", input_name);

        let input_tensor = ort::Value::from_array(batched.view())?;
        let inputs: Vec<(String, ort::SessionInputValue)> =
            vec![(input_name, ort::SessionInputValue::from(input_tensor))];

        let outputs = self.model.run(inputs)?;

        let embedding: Array2<f32> = outputs[0]
            .try_extract_tensor::<f32>()?
            .into_dimensionality()
            .context("Embedding shape conversion failed")?
            .to_owned();

        let logits: Array2<f32> = outputs[1]
            .try_extract_tensor::<f32>()?
            .into_dimensionality()
            .context("Logits shape conversion failed")?
            .to_owned();

        let class_idx = logits
            .row(0)
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let predicted = self
            .metadata
            .keys
            .get(&class_idx.to_string())
            .and_then(|v| v.get("label"))
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => "unknown".to_string(),
            })
            .unwrap_or_else(|| "unknown".to_string());

        println!("Predicted class: {}", predicted);

        let embed_vec = embedding.row(0).to_owned();
        let top_matches = self.find_top_k(embed_vec, 5);

        Ok(top_matches)
    }

    fn find_top_k(&self, query: Array1<f32>, k: usize) -> Vec<(String, f32)> {
        let query_norm = query.dot(&query).sqrt();
        let db_norms = self.embeddings.map_axis(Axis(1), |v| v.dot(&v).sqrt());

        let sims: Vec<f32> = self
            .embeddings
            .outer_iter()
            .zip(db_norms)
            .map(|(db_vec, db_norm)| query.dot(&db_vec) / (query_norm * db_norm))
            .collect();

        let mut idxs: Vec<usize> = (0..sims.len()).collect();
        idxs.sort_unstable_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap());

        idxs.into_iter()
            .take(k)
            .map(|i| {
                let label = self
                    .metadata
                    .keys
                    .get(&self.metadata.internal_ids[i].to_string())
                    .and_then(|v| v.get("label"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                (label, sims[i])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_fish_classifier() {
        let classifier = FishClassifier::new().expect("Failed to create classifier");
        let dummy_input = Array3::<f32>::zeros((3, 224, 224));

        let result = classifier
            .classify(dummy_input)
            .expect("Classification failed");

        println!("Classification results: {:?}", result);
        assert!(!result.is_empty(), "Should return at least one result");
    }
}

struct KeyVisitor;

impl<'de> Visitor<'de> for KeyVisitor {
    type Value = HashMap<String, serde_json::Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a map with string or number keys")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        let mut hm = HashMap::new();
        while let Some((key, value)) = map.next_entry()? {
            let key_str = match key {
                serde_json::Value::String(s) => s,
                serde_json::Value::Number(n) => n.to_string(),
                _ => return Err(DeError::custom("Invalid key type in keys map")),
            };
            hm.insert(key_str, value);
        }
        Ok(hm)
    }
}

fn deserialize_keys_int_to_string<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_map(KeyVisitor)
}
