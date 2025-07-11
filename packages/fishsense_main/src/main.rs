use fish_logic::db::search_similar_fish;
use fish_logic::fish::FishEmbeddingExtractor;
use std::error::Error;
use std::env;
use image::{ImageReader, DynamicImage};
use ndarray::Array3;

// Unused imports (commented out since functions are not used):
// use fish_logic::db::{get_database_stats, ingest_data};

fn preprocess_image(img: DynamicImage) -> Array3<f32> { // preprocessing for embedding extraction.
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);  // Resize the image to 224x224 pixels using Lanczos3 filter
    // this can result in the image being stretched or squished. Does not maintain aspect ratio. Need to look into if this matters? 
    let img_rgb = img.to_rgb8(); // into rbg 
    let mut array = Array3::<f32>::zeros((3, 224, 224)); // 3D array with dimensions (channels, height, width)
    for (y, row) in img_rgb.rows().enumerate() { // Iterate over each pixel in the image  and normalized to a range of [0.0, 1.0].
        for (x, pixel) in row.enumerate() {
            array[[0, y, x]] = pixel[0] as f32 / 255.0; // R
            array[[1, y, x]] = pixel[1] as f32 / 255.0; // G
            array[[2, y, x]] = pixel[2] as f32 / 255.0; // B
        }
    } 
    array
}

async fn identify_fish(image_path: &str) -> Result<(), Box<dyn Error>> { // identifies a fish species by its embedding and vector search.
    println!("Fish Species Identification");
    println!("{}", image_path);  //image path
    let img = ImageReader::open(image_path)?.decode()?;
    println!("dimensions: {}x{}", img.width(), img.height());

    let preprocessed = preprocess_image(img); // call preprocess_image 
    println!("Preprocessed image shape: {:?}", preprocessed.dim()); // Evaluate: how does this look like?

    let extractor = FishEmbeddingExtractor::new().await?; // Initialize the embedding extractor
    println!("extractor initialized");

    let embedding_vector = extractor.extract_embedding(preprocessed)?;  //  Extract the embedding vector
    println!("{}-dimensional embedding vector", embedding_vector.len());

    println!("Searching the database for similar fish..."); // Search the database for similar embeddings
    let similar_fish = search_similar_fish(embedding_vector, 10).await?; // gets 10 results
    
    println!("\nTop matches:");
    for (i, (label, distance, id)) in similar_fish.iter().enumerate() {
        let similarity = ((200.0 - distance) / 200.0 * 100.0).max(0.0);
        println!(
            "{}. {} ({:.1}% match, distance: {:.2}, id: {})",
            i + 1,
            label,
            similarity,
            distance,
            id
        );
    }

    if let Some((top_species, _, _)) = similar_fish.first() { // Display the best match
        println!("\nBest match: {}", top_species);
    }

//     let unique_species: std::collections::HashSet<String> = similar_fish // Count the number of unique species in vector search
//         .iter()
//         .map(|(species, _, _)| species.clone())
//         .collect();
//     println!("\nFound {} unique species in the top 10 matches.", unique_species.len());

    Ok(())
}

// async fn setup_database() -> Result<(), Box<dyn Error>> { // Sets up the fish database by ingesting data.
//     println!("Setting up the fish database...");
//     ingest_data().await?; // Call the ingest_data function 
//     println!("Process is complete.");
//     Ok(())
// }

async fn show_database_info() -> Result<(), Box<dyn Error>> { // Displays information about the fish database
    println!("FishSense - Database");
    get_database_stats().await?; // calling the get_database_stats function to display database stats
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: cargo run <image_path>");
        return Ok(());
    }
    let image_path = &args[1];
    if std::path::Path::new(image_path).exists() {
        identify_fish(image_path).await?;
    } else {
        println!("'{}' not found", image_path);
    } Ok(())
}