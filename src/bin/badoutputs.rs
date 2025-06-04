use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    collections::HashMap,
};

use anyhow::{Context, Result};
use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use ndarray::Array1;
use serde::Serialize;
use fishsense::fish::{FishHeadTailDetector, HeadTailError};
use std::panic::{self, AssertUnwindSafe};

#[derive(Serialize)]
struct Point {
    x: usize,
    y: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ResultData {
    Success { snout: Point, fork: Point },
    Error { error: String },
}

#[derive(Serialize)]
struct ImageResult {
    id: String,
    result: ResultData,
}

fn main() -> Result<()> {
    let input_dir = "./data/segmented";
    let truthy_dir = "./data/truthy";
    let output_dir = "./data/badoutputs";

    fs::create_dir_all(output_dir)?;

    // Step 1: collect base filenames from truthy_dir
    let valid_names: std::collections::HashSet<String> = fs::read_dir(truthy_dir)?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.path().file_stem()?.to_str().map(|s| s.to_string()))
        .collect();

    let mut results = Vec::new();

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip non-jpg/jpeg files
        let ext = path.extension().map(|ext| ext.to_ascii_lowercase());
        if ext != Some("jpg".into()) && ext != Some("jpeg".into()) {
            continue;
        }

        // Get base filename and check if it's in valid_names
        let file_stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        if !valid_names.contains(&file_stem) {
            continue;
        }

        println!("Processing {}", file_stem);

        // Load
        let img = ImageReader::open(&path)
            .with_context(|| format!("Failed to open image {:?}", path))?
            .decode()
            .with_context(|| format!("Failed to decode image {:?}", path))?
            .to_luma8();

        let mut img_mut = img.clone();

        // Detect
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            FishHeadTailDetector::find_head_tail(&mut img_mut)
        }));
        let result = match result {
            Ok(inner_res) => inner_res,
            Err(_) => Err(HeadTailError::PolygonError),
        };

        // Save output image
        let output_path = Path::new(output_dir).join(format!("{}_out.jpg", file_stem));
        img_mut.save(&output_path)
            .with_context(|| format!("Failed to save output image {:?}", output_path))?;

        // Create JSON entry
        let json_entry = match result {
            Ok((tail, head)) => ImageResult {
                id: file_stem.clone(),
                result: ResultData::Success {
                    snout: Point { x: head[0], y: head[1] },
                    fork: Point { x: tail[0], y: tail[1] },
                },
            },
            Err(e) => ImageResult {
                id: file_stem.clone(),
                result: ResultData::Error {
                    error: format!("{:?}", e),
                },
            },
        };

        results.push(json_entry);
    }

    // Save JSON file
    let json_path = Path::new(output_dir).join("badresults.json");
    let mut json_file = File::create(&json_path)?;
    let json_string = serde_json::to_string_pretty(&results)?;
    json_file.write_all(json_string.as_bytes())?;

    println!("Processing complete. Results saved to {:?}", json_path);
    Ok(())
}