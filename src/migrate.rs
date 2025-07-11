use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use lancedb::{connect, Connection};
use serde_json::Value;
use arrow_array::{
    Float32Array, StringArray, UInt32Array, FixedSizeListArray,
    RecordBatch, RecordBatchIterator
};
use arrow_schema::{DataType, Field, Schema};
use arrow_array::types::Float32Type;

const EMBEDDING_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/embeddings.npy?download=true";
const METADATA_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/database.json?download=true";

pub async fn run_migration() -> Result<()> {
    let uri = "data/fish_lancedb";
    let db = connect(uri).execute().await?;
    
    // Check if table exists and has data
    match db.open_table("fish_embeddings").execute().await {
        Ok(table) => {
            let count = table.count_rows(None).await?;
            if count > 0 {
                println!("Database already exists with {} embeddings", count);
                return Ok(());
            }
        }
        Err(_) => {}
    }
    
    // Download files if needed
    if !Path::new("embeddings.npy").exists() {
        println!("Downloading embeddings...");
        download_file(EMBEDDING_URL, "embeddings.npy").await?;
    }
    
    if !Path::new("database.json").exists() {
        println!("Downloading metadata...");
        download_file(METADATA_URL, "database.json").await?;
    }
    
    // Load the fish data
    println!("Loading embeddings and metadata...");
    let embeddings = load_fish_embeddings()?;
    let metadata = load_fish_metadata()?;
    
    println!("Loaded {} embeddings and {} species classes", 
             embeddings.len(), metadata.keys.len());
    
    // Create the fish vector database
    create_fish_database(&db, &embeddings, &metadata).await?;
    
    println!("Migration completed successfully");
    Ok(())
}

#[derive(serde::Deserialize)]
struct FishMetadata {
    internal_ids: Vec<usize>,
    keys: HashMap<String, serde_json::Value>,
}

fn load_fish_embeddings() -> Result<Vec<Vec<f32>>> {
    use ndarray::Array2;
    use ndarray_npy::read_npy;
    
    let embeddings: Array2<f32> = read_npy("embeddings.npy")?;
    println!("Embedding shape: {:?}", embeddings.dim());
    Ok(embeddings.outer_iter().map(|row| row.to_vec()).collect())
}

fn load_fish_metadata() -> Result<FishMetadata> {
    let content = std::fs::read_to_string("database.json")?;
    let metadata: FishMetadata = serde_json::from_str(&content)?;
    Ok(metadata)
}

async fn create_fish_database(
    db: &Connection,
    embeddings: &[Vec<f32>],
    metadata: &FishMetadata,
) -> Result<()> {
    
    let vector_dim = embeddings.get(0).map(|e| e.len()).unwrap_or(512) as i32;
    println!("Vector dimension: {}", vector_dim);
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("embedding_id", DataType::UInt32, false),
        Field::new("class_id", DataType::UInt32, false),
        Field::new("species_name", DataType::Utf8, false),
        Field::new("species_id", DataType::Utf8, false),
        Field::new(
            "embedding", 
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)), 
                vector_dim
            ), 
            false
        ),
    ]));
    
    let mut embedding_ids = Vec::new();
    let mut class_ids = Vec::new();
    let mut species_names = Vec::new();
    let mut species_ids = Vec::new();
    let mut fish_embeddings = Vec::new();
    
    // Use the original mapping from internal_ids and normalize vectors
    for (embedding_idx, embedding) in embeddings.iter().enumerate() {
        // Get the class ID for this embedding from internal_ids
        let class_id = if embedding_idx < metadata.internal_ids.len() {
            metadata.internal_ids[embedding_idx] as u32
        } else {
            continue;
        };
        
        // Normalize the embedding vector for proper cosine distance
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized_embedding = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect::<Vec<f32>>()
        } else {
            embedding.clone()
        };
        
        // Look up species info from keys using class_id
        if let Some(class_info) = metadata.keys.get(&class_id.to_string()) {
            let species_name = class_info
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("Unknown Species")
                .to_string();
            
            let species_uuid = class_info
                .get("species_id")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown-uuid")
                .to_string();
            
            embedding_ids.push(embedding_idx as u32);
            class_ids.push(class_id);
            species_names.push(species_name);
            species_ids.push(species_uuid);
            fish_embeddings.push(Some(normalized_embedding));
        }
    }
    
    println!("Processed {} valid embeddings", embedding_ids.len());
    
    let vector_array = Arc::new(
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            fish_embeddings.into_iter().map(|opt_vec| {
                opt_vec.map(|vec| vec.into_iter().map(Some).collect::<Vec<Option<f32>>>())
            }),
            vector_dim,
        ),
    );
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(embedding_ids)),
            Arc::new(UInt32Array::from(class_ids)),
            Arc::new(StringArray::from(species_names)),
            Arc::new(StringArray::from(species_ids)),
            vector_array,
        ],
    )?;
    
    let batches = RecordBatchIterator::new(
        vec![batch].into_iter().map(Ok),
        schema.clone(),
    );
    
    println!("Creating table and index...");
    let table = db
        .create_table("fish_embeddings", Box::new(batches))
        .execute()
        .await?;
    
    // Create vector index for fast similarity search
    match table.create_index(&["embedding"], lancedb::index::Index::Auto).execute().await {
        Ok(_) => println!("Vector index created"),
        Err(e) => println!("Index creation failed (table still usable): {}", e),
    }
    
    // Verify the migration
    let count = table.count_rows(None).await?;
    println!("Database created with {} embeddings ready for search", count);
    
    Ok(())
}

async fn download_file(url: &str, path: &str) -> Result<()> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    std::fs::write(path, bytes)?;
    Ok(())
}