use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use image::{GrayImage, Luma};
use ndarray::{Array2, Array3};
use serde::Deserialize;
use std::fs;

use fishsense::fish::FishSegmentation;

#[derive(Deserialize)]
struct Detection {
    id: u32,
    img: String,
    head: Option<[u32; 2]>,
    tail: Option<[u32; 2]>,
}

fn main() {
    let json_path = "./data/cleaned_data.json";
    let output_dir = "./data/segmented";

    let file = File::open(json_path).expect("Failed to open JSON file");
    let reader = BufReader::new(file);
    let detections: Vec<Detection> = serde_json::from_reader(reader).expect("Invalid JSON");

    for detection in detections {
        println!("Processing ID {}", detection.id);
        if let Ok(img_path) = download_image(&detection.img, detection.id) {
            segment_and_save(
                &img_path,
                output_dir,
                detection.id,
                detection.head.map(|h| (h[0], h[1])),
                detection.tail.map(|t| (t[0], t[1])),
            );
        } else {
            eprintln!("Failed to download image for ID {}", detection.id);
        }
    }
}

fn download_image(url: &str, id: u32) -> Result<String, reqwest::Error> {
    let response = reqwest::blocking::get(url)?;
    let bytes = response.bytes()?;

    let file_path = format!("./data/{}.jpeg", id);
    std::fs::write(&file_path, &bytes).expect("Failed to write image to disk");
    Ok(file_path)
}

fn segment_and_save(
    input_path: &str,
    output_dir: &str,
    image_id: u32,
    head: Option<(u32, u32)>,
    tail: Option<(u32, u32)>
) {
    let img = image::ImageReader::open(input_path).unwrap().decode().unwrap().to_rgb8();
    let (width, height) = img.dimensions();

    let img_array: Array3<u8> = Array3::from_shape_fn((height as usize, width as usize, 3), |(y, x, c)| {
        img.get_pixel(x as u32, y as u32)[c]
    });

    let mut seg = FishSegmentation::from_web().unwrap();
    seg.load_model().unwrap();
    let segmentations: Array2<i32> = seg.inference(&img_array).unwrap().mapv(|v| v as i32);

    let mut unique_labels: HashSet<i32> = segmentations.iter().cloned().collect();
    unique_labels.remove(&0); // Remove background

    for &label in &unique_labels {
        let mask = segmentations.mapv(|v| v == label);

        if !mask.iter().any(|&b| b) {
            continue;
        }

        let mut special = false;

        if let Some((hx, hy)) = head {
            if hx < width && hy < height && mask[[hy as usize, hx as usize]] {
                special = true;
            }
        }

        if let Some((tx, ty)) = tail {
            if tx < width && ty < height && mask[[ty as usize, tx as usize]] {
                special = true;
            }
        }

        let mut output = GrayImage::new(width, height);
        for ((y, x), is_fg) in mask.indexed_iter() {
            let val = if *is_fg { 255 } else { 0 };
            output.put_pixel(x as u32, y as u32, Luma([val]));
        }

        let file_name = if special {
            format!("{}/{}_fish{}_s.jpeg", output_dir, image_id, label)
        } else {
            format!("{}/{}_fish{}.jpeg", output_dir, image_id, label)
        };
        output.save(&file_name).unwrap();
    }
    fs::remove_file(input_path).unwrap();
}