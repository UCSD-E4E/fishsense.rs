use image::GenericImageView;
use ndarray::Array3;
use std::env;
use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use lancedb::connect;
use fishsense::fish::VectorDbFishClassifier;

mod migrate;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("FishSense Vector Database Classification System");
    println!("Process ID: {}", std::process::id());
    
    let uri = "data/fish_lancedb"; 
    let db = connect(uri).execute().await?;
    
    // Check if migration is needed
    match db.open_table("fish_embeddings").execute().await {
        Ok(table) => {
            let count = table.count_rows(None).await?;
            if count > 0 {
                println!("Database ready with {} embeddings", count);
            } else {
                println!("Database exists but empty, running migration...");
                migrate::run_migration().await?;
            }
        },
        Err(_) => {
            println!("Database not found, running migration...");
            migrate::run_migration().await?;
        }
    }
    
    // Get image path from command-line args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release -- path/to/image.jpg");
        std::process::exit(1);
    }
    let image_path = &args[1];
    
    // Initialize classifier
    let init_start = Instant::now();
    let classifier = VectorDbFishClassifier::new().await?;
    let init_time = init_start.elapsed();
    
    // Load and process image
    let array = load_and_process_image(image_path)?;
    
    // Run classification
    let classify_start = Instant::now();
    let predictions = classifier.classify(array).await?;
    let classify_time = classify_start.elapsed();
    
    // Print results
    println!("\nFish Classification Results:");
    for (label, score) in predictions {
        println!("{} ({:.3})", label, score);
    }
    
    println!("\nPerformance:");
    println!("Initialization: {:?}", init_time);
    println!("Classification: {:?}", classify_time);
    println!("Total: {:?}", init_time + classify_time);
    
    Ok(())
}

fn load_and_process_image(image_path: &str) -> Result<Array3<f32>> {
    let img = image::open(&Path::new(image_path))?
        .resize_exact(224, 224, image::imageops::FilterType::Nearest);
    
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
    
    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[2, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
    }
    
    Ok(array)
}