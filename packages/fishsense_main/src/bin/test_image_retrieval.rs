use fish_logic::db::{get_database_stats, search_similar_fish};
use fish_logic::fish::FishEmbeddingExtractor;
use std::error::Error;
use image::{ImageReader, DynamicImage};
use ndarray::Array3;
use std::env;
use std::path::Path;

fn preprocess_image(img: DynamicImage) -> Array3<f32> {
    // Resize image to 224x224 (standard input size for most models)
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let img_rgb = img.to_rgb8();
    // Convert to Array3<f32> with shape (C, H, W) and normalize to [0, 1]
    let mut array = Array3::<f32>::zeros((3, 224, 224));
    
    for (y, row) in img_rgb.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            // Normalize pixel values to [0, 1] range
            array[[0, y, x]] = pixel[0] as f32 / 255.0; // R
            array[[1, y, x]] = pixel[1] as f32 / 255.0; // G
            array[[2, y, x]] = pixel[2] as f32 / 255.0; // B
        }
    } array
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🐟 Testing image-based fish retrieval...");
    
    // Get database statistics first
    get_database_stats().await?;
    
    let args: Vec<String> = env::args().collect();
    // Check that the user provided a path
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }
    // Get the image path
    let img_path = &args[1];
    // Optional: check if the path exists
    if !Path::new(img_path).exists() {
        eprintln!("File does not exist: {}", img_path);
        std::process::exit(1);
    }

    println!("Using image path: {}", img_path);

    let img = ImageReader::open(img_path)?
        .decode()
        .map_err(|e| format!("Failed to decode image: {}", e))?;
    
    println!("Original image size: {}x{}", img.width(), img.height());
    
    let preprocessed = preprocess_image(img);
    println!("Preprocessed image shape: {:?}", preprocessed.dim());
    
    // Initialize the fish embedding extractor
    println!("\n🧠 Initializing FishEmbeddingExtractor...");
    let extractor = FishEmbeddingExtractor::new().await
        .map_err(|e| format!("Failed to initialize embedding extractor: {}", e))?;
    
    if !extractor.is_ready() {
        return Err("Embedding extractor is not ready".into());
    }
    // Extract embedding from the image
    println!("\n🔍 Extracting embedding from fish image...");
    match extractor.extract_embedding(preprocessed) {
        Ok(embedding_vector) => {
            println!("Extracted embedding vector (length: {})", embedding_vector.len());
            // Search for similar fish using the embedding
            println!("\n Searching for similar fish in database.");
            
            match search_similar_fish(embedding_vector, 10).await {
                Ok(similar_fish) => {
                    println!("Top 10 most similar fish from database:");
                    for (i, (label, distance, id)) in similar_fish.iter().enumerate() {
                        println!("   {}. {} (distance: {:.4}, id: {})", i + 1, label, distance, id);
                    }
                    if let Some((top_match, distance, id)) = similar_fish.first() {
                        println!("\nBest match: {} (distance: {:.4}, id: {})", top_match, distance, id);
                        
                        if *distance < 0.5 {
                            println!("High confidence match (distance < 0.5)");
                        } else if *distance < 1.0 {
                            println!("Moderate confidence match (distance < 1.0)");
                        } else {
                            println!("Low confidence match (distance >= 1.0)");
                        }
                    }
                }
                Err(e) => {
                    println!("Database search failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Embedding extraction failed: {}", e);
            println!("This might be due to model compatibility issues or preprocessing differences.");
        }
    }
    println!("\nTest completed!");
    Ok(())
}
