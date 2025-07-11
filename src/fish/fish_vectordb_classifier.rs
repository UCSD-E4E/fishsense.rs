use app_dirs2::{app_root, AppDataType, AppInfo};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use ort::Session;
use reqwest::blocking::get;
use std::fs::{create_dir_all, File};
use std::io::copy;
use std::path::PathBuf;
use std::time::Instant;
use std::collections::HashMap;
use serde::Deserialize;
use serde_json::Value;

// LanceDB imports
use lancedb::{connect, Connection, Table};
use arrow_array::{StringArray, Float32Array, UInt32Array};
use futures::TryStreamExt;
use lancedb::query::{QueryBase, ExecutableQuery};
use lancedb::DistanceType;

const MODEL_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/fish_classifier.onnx?download=true";
const METADATA_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/database.json?download=true";

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
    pub keys: HashMap<String, serde_json::Value>,
}

pub struct VectorDbFishClassifier {
    model: Session,
    fish_table: Table,
}

impl VectorDbFishClassifier {
    pub async fn new() -> Result<Self> {
        let model_path = Self::download_file("fish_classifier.onnx", MODEL_URL)?;
        let model = Session::builder()?.commit_from_file(&model_path)?;
        
        println!("Model loaded successfully");
        println!("Model inputs:");
        for inp in &model.inputs {
            println!("- Name: {}, Type: {:?}", inp.name, inp.input_type);
        }
        
        let uri = "data/fish_lancedb";
        let db = connect(uri).execute().await?;
        
        let fish_table = db.open_table("fish_embeddings").execute().await
            .context("Failed to open fish_embeddings table. Run migration first!")?;
        
        let species_count = fish_table.count_rows(None).await?;
        println!("Connected to database with {} embeddings", species_count);
        
        Ok(VectorDbFishClassifier {
            model,
            fish_table,
        })
    }

    fn download_file(filename: &str, url: &str) -> Result<PathBuf> {
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

    pub async fn classify(&self, input_tensor: Array3<f32>) -> Result<Vec<(String, f32)>> {
        // Run ONNX model to get embedding
        let batched = input_tensor.insert_axis(Axis(0));
        let input_name = self.model.inputs[0].name.clone();
        
        let input_tensor = ort::Value::from_array(batched.view())?;
        let inputs: Vec<(String, ort::SessionInputValue)> = vec![
            (input_name, ort::SessionInputValue::from(input_tensor))
        ];

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

        // Get direct classification from logits for comparison
        let class_idx = logits
            .row(0)
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let predicted_class_name = self.get_class_name_from_logits(class_idx).await?;
        println!("Direct prediction: {}", predicted_class_name);

        // Use vector database search for top-k results
        let embed_vec = embedding.row(0).to_owned();
        let top_matches = self.vector_search(embed_vec, 5).await?;
        
        Ok(top_matches)
    }

    async fn get_class_name_from_logits(&self, class_idx: usize) -> Result<String> {
        let meta_path = Self::download_file("database.json", METADATA_URL)?;
        let metadata_content = std::fs::read_to_string(&meta_path)?;
        let json: Value = serde_json::from_str(&metadata_content)?;
        
        if let Some(keys) = json.get("keys").and_then(|k| k.as_object()) {
            if let Some(value) = keys.get(&class_idx.to_string()) {
                if let Some(label) = value.get("label").and_then(|l| l.as_str()) {
                    return Ok(label.to_string());
                }
            }
        }
        
        Ok(format!("unknown_class_{}", class_idx))
    }

    async fn vector_search(&self, query_embedding: Array1<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        let query_vector: Vec<f32> = query_embedding.to_vec();
        
        // Normalize the query vector for proper cosine distance
        let norm: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized_query = if norm > 0.0 {
            query_vector.iter().map(|x| x / norm).collect::<Vec<f32>>()
        } else {
            query_vector
        };
        
        let start = Instant::now();
        
        let mut results = self.fish_table
            .query()
            .nearest_to(normalized_query)?
            .distance_type(DistanceType::Cosine)
            .limit(k)
            .execute()
            .await?;
        
        let search_time = start.elapsed();
        
        let mut fish_matches = Vec::new();
        
        while let Some(batch) = results.try_next().await? {
            let distances = batch
                .column_by_name("_distance")
                .context("Missing _distance column from LanceDB results")?
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast _distance column")?;
                
            let species_names = batch
                .column_by_name("species_name")
                .context("Missing species_name column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to cast species_name column")?;
            
            let class_ids = batch
                .column_by_name("class_id")
                .context("Missing class_id column")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Failed to cast class_id column")?;
            
            for i in 0..batch.num_rows() {
                let distance = distances.value(i);
                let species_name = species_names.value(i).to_string();
                let class_id = class_ids.value(i);
                
                // Convert cosine distance to confidence score
                let confidence = (1.0 - (distance / 2.0)).clamp(0.0, 1.0);
                
                fish_matches.push((species_name, confidence));
                
                if i == 0 {
                    println!("Top match: {} (class_id: {}, distance: {:.4}, confidence: {:.3})", 
                             species_names.value(i), class_id, distance, confidence);
                }
            }
        }
        
        println!("Vector search completed in {:?}", search_time);
        
        Ok(fish_matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[tokio::test]
    async fn test_vector_db_fish_classifier() {
        let classifier = VectorDbFishClassifier::new().await.expect("Failed to create classifier");
        let dummy_input = Array3::<f32>::zeros((3, 224, 224));
    
        let result = classifier.classify(dummy_input).await.expect("Classification failed");

        println!("Classification results: {:?}", result);
        assert!(!result.is_empty(), "Should return at least one result");
        
        for (_, confidence) in &result {
            assert!(confidence >= &0.0 && confidence <= &1.0, 
                   "Confidence score should be between 0 and 1, got {}", confidence);
        }
    }
}