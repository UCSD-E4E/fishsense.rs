use arrow::array::{Float32Array, StringArray};
use lancedb::connect;
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::error::Error;
use futures::stream::StreamExt;

// Imports for ingestion (currently commented out):
// use arrow::array::{FixedSizeListArray, RecordBatch};
// use arrow::datatypes::{DataType, Field, Schema};
// use arrow::record_batch::RecordBatchIterator;
// use lance_arrow::FixedSizeListArrayExt;
// use npyz::NpyFile;
// use serde::Deserialize;
// use std::collections::HashMap;
// use std::sync::Arc;


// Data ingestion is currently disabled 
// Uncomment the sections below to re-enable
// const EMBEDDING_URL: &str =
//     "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/embeddings.npy?download=true";
// const METADATA_URL: &str =
//     "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/database.json?download=true";

// const VECTOR_DIM: i32 = 128;

/*
#[derive(Debug, Deserialize)] // structure from fish_embedding_extractor.rs
struct Metadata {
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
*/
pub async fn get_table() -> Result<Table, Box<dyn Error>> {
    let db = connect("/tmp/lancedb").execute().await?;
    let tbl = db.open_table("fish_embeddings").execute().await?;
    Ok(tbl)
}
/*

pub async fn ingest_data() -> Result<(), Box<dyn Error>> {
    println!(" Starting data ingestion process");

    let db = connect("/tmp/lancedb").execute().await?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("vector", DataType::new_fixed_size_list(DataType::Float32, VECTOR_DIM, false), false),
        Field::new("label", DataType::Utf8, false),
        Field::new("embedding_id", DataType::UInt64, false),
    ]));

    let _ = db.drop_table("fish_embeddings").await;
    let tbl = db.create_empty_table("fish_embeddings", schema).execute().await?;
    println!("'fish_embeddings' is ready.");

    println!("Downloading data");
    let client = reqwest::Client::new();
    let embedding_bytes = client.get(EMBEDDING_URL).send().await?.bytes().await?;
    let metadata_bytes = client.get(METADATA_URL).send().await?.bytes().await?;
    println!("Data downloaded.");

    println!("Parsing data");
    let npy_file = NpyFile::new(&embedding_bytes[..])?;
    let embeddings: Vec<f32> = npy_file.into_vec::<f32>()?;

    let metadata: Metadata = serde_json::from_slice(&metadata_bytes)?; // Parse the metadata as a single JSON object

    let num_embeddings = embeddings.len() / VECTOR_DIM as usize; // Debug information
    let num_internal_ids = metadata.internal_ids.len();
    let num_keys = metadata.keys.len();
    
    println!(" Data analysis:");
    println!(" Embeddings: {}", num_embeddings);
    println!(" Internal IDs: {}", num_internal_ids);
    println!(" Keys: {}", num_keys);
    println!(" Ratio: {:.1}:1 (internal_ids:embeddings)", num_internal_ids as f64 / num_embeddings as f64);
    
    // Create a mapping from unique internal_id values to labels
    println!(" Building internal_id to label mapping");
    let mut id_to_label: HashMap<usize, String> = HashMap::new();

    for (key_str, value) in &metadata.keys {
        if let Ok(id) = key_str.parse::<usize>() {
            if let Some(label) = value.get("label").and_then(|l| l.as_str()) {
                id_to_label.insert(id, label.to_string());
            }
        }
    }

    println!("Built mapping for {} unique fish IDs", id_to_label.len());
    // For each embedding, we need to determine which internal_id it corresponds to
    // Since we have more internal_ids than embeddings, we'll map each embedding index
    // to the corresponding internal_id at that index
    let mut labels = Vec::new();
    let mut embedding_ids = Vec::new();
    
    // Group internal_ids by their position to find the mapping pattern
    let groups_per_embedding = num_internal_ids / num_embeddings;
    println!(" Detected {} internal_ids per embedding", groups_per_embedding);
    
    for i in 0..num_embeddings {
        // Take the first internal_id from each group
        let internal_id_index = i * groups_per_embedding;
        let internal_id = metadata.internal_ids[internal_id_index];
        
        // Look up the label using the internal_id
        let label = id_to_label.get(&internal_id)
            .cloned()
            .unwrap_or_else(|| {
                // Fallback: try using the embedding index directly
                id_to_label.get(&i).cloned().unwrap_or("unknown".to_string())
            });
        
        labels.push(label);
        embedding_ids.push(i as u64);
    }
    
    // Count unique labels for verification
    let mut unique_labels = std::collections::HashSet::new();
    for label in &labels {
        unique_labels.insert(label.clone());
    }
    println!(" Found {} unique fish species/labels", unique_labels.len());
    
    println!(" Data parsed and aligned using grouped internal_id mapping.");

    println!(" Inserting data into LanceDB...");
    let vector_column = Arc::new(FixedSizeListArray::try_new_from_values(
        Float32Array::from(embeddings),
        VECTOR_DIM,
    )?);
    let label_column = Arc::new(StringArray::from(labels));
    let id_column = Arc::new(arrow::array::UInt64Array::from(embedding_ids));

    let batch = RecordBatch::try_new(
        tbl.schema().await?.clone(), 
        vec![vector_column, label_column, id_column]
    )?;

    let reader = RecordBatchIterator::new(vec![Ok(batch)], tbl.schema().await?.clone());
    tbl.add(Box::new(reader)).execute().await?;

    let row_count = tbl.count_rows(None).await?;
    println!("Success! Inserted {} rows.", row_count);

    Ok(())
}
*/

pub async fn search_similar_fish(query_vector: Vec<f32>, limit: usize) -> Result<Vec<(String, f64, u64)>, Box<dyn Error>> {
    let tbl = get_table().await?;

    let mut results = tbl
        .vector_search(query_vector)?
        .limit(limit)
        .execute()
        .await?;

    let mut fish_results = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;
        let labels = batch.column_by_name("label").unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ids = batch.column_by_name("embedding_id").unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        let distances = batch.column_by_name("_distance").unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        
        for i in 0..batch.num_rows() {
            let label = labels.value(i).to_string();
            let distance = distances.value(i) as f64;
            let id = ids.value(i);
            
            fish_results.push((label, distance, id));
        }
    } Ok(fish_results)
}

pub async fn get_all_fish_labels() -> Result<Vec<String>, Box<dyn Error>> {
    let tbl = get_table().await?;
    let mut results = tbl
        .query()
        .select(Select::Columns(vec!["label".to_string()]))
        .execute()
        .await?;
    
    let mut labels = std::collections::HashSet::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;
        let label_array = batch.column_by_name("label").unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        
        for i in 0..batch.num_rows() {
            labels.insert(label_array.value(i).to_string());
        }
    }
    Ok(labels.into_iter().collect())
}

pub async fn get_database_stats() -> Result<(), Box<dyn Error>> {
    let tbl = get_table().await?;

    let total_rows = tbl.count_rows(None).await?;
    println!("Database Statistics:");
    println!("Total fish embeddings: {}", total_rows);
    let labels = get_all_fish_labels().await?;
    println!("Unique fish species: {}", labels.len());
    println!("Sample species:");
    for (i, label) in labels.iter().take(10).enumerate() {
        println!("{}. {}", i + 1, label);
    }
    if labels.len() > 10 {
        println!("and {} more", labels.len() - 10);
    }
    Ok(())
}