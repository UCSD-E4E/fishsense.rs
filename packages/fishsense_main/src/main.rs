use fish_logic::fish::FishClassifier;
// use image::GenericImageView;
use ndarray::Array3;
use std::env;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Get image path from command-line args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release -- path/to/image.jpg");
        std::process::exit(1);
    }
    let image_path = &args[1];

    // Load the image
    let img = image::open(&Path::new(image_path))
        .expect("Failed to open image")
        .resize_exact(224, 224, image::imageops::FilterType::Nearest);

    // Convert image to RGB and ndarray::Array3<f32>
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[2, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
    }

    // Run classifier
    let classifier = FishClassifier::new().await()?;
    let predictions = classifier.classify(array).await()?;

    // Print results
    println!("\n Fish Classification Results:");
    for (label, score) in predictions {
        println!("{} ({:.2})", label, score);
    }

    Ok(())
}
