use anyhow::{Context, Result};
use app_dirs2::{AppDataType, AppInfo, app_root};
use ndarray::{Array3, Axis};
use ort::Session;
use std::fs::create_dir_all;
use std::path::PathBuf;

const MODEL_URL: &str = "https://huggingface.co/unushri/onnx-fish-classifier/resolve/main/fish_classifier.onnx?download=true";

pub struct FishEmbeddingExtractor {
    model: Session, // ONNX runtime session 
}

impl FishEmbeddingExtractor {
    
    pub async fn new() -> Result<Self> { // Creates a new instance of the FishEmbeddingExtractor
        let model_path = Self::download_file("fish_classifier.onnx", MODEL_URL).await?;

        let model = Session::builder()?  // loading the ONNX runtime session (the model)
            .commit_from_file(&model_path)?;

        println!("Model loaded for embedding extraction:");
        for input in &model.inputs {
            println!("Input: {}, Type: {:?}", input.name, input.input_type);
        }
        for output in &model.outputs {
            println!("Output: {}, Type: {:?}", output.name, output.output_type);
        }
        Ok(FishEmbeddingExtractor { model })
    }

    // pub fn is_ready(&self) -> bool { // Checks if the extractor is ready to use
    //     true // Always returns true for now
    // }

    /// Downloads a file from a URL and saves it to the local cache directory
    async fn download_file(filename: &str, url: &str) -> Result<PathBuf> {
        // Determine the cache directory for storing the model
        let mut path = app_root(
            AppDataType::UserCache,
            &AppInfo {
                name: "fishsense",
                author: "unushri",
            },
        )?;
        path.push("models"); // Add a "models" subdirectory
        create_dir_all(&path)?; // Create the directory if it doesn't exist
        path.push(filename); // Add the filename to the path

        
        if !path.exists() { // Check if the file already exists
            println!("Downloading {} from {}", filename, url);
            let response = reqwest::get(url).await?; // Download the file 
            let bytes = response.bytes().await?;
            std::fs::write(&path, &bytes)?; // Save the downloaded file 
        }
        // Return the path 
        Ok(path)
    }

    pub fn extract_embedding(&self, input_tensor: Array3<f32>) -> Result<Vec<f32>> { // Extracts a 128-dimensional embedding vector from an input image
        // Add a batch dimension to the input tensor
        // Input shape changes from (3, 224, 224) to (1, 3, 224, 224)
        let batched_tensor = input_tensor.insert_axis(Axis(0));

        // Get the name of the input tensor expected by the model
        let input_name = self.model.inputs[0].name.clone();
        println!("Extracting embedding using input: {}", input_name);

        // Convert the input tensor into a format compatible with ONNX runtime
        let input_tensor = ort::Value::from_array(batched_tensor.view())?;
        let inputs = vec![
            (input_name, ort::SessionInputValue::from(input_tensor))
        ];

        let outputs = self.model.run(inputs)?; // Run the model inference

        let embedding = outputs[0] // Extract the embedding from the first output of the model
            .try_extract_tensor::<f32>()? // Extract the tensor as f32 values
            .into_dimensionality::<ndarray::Ix2>() // Ensure it's a 2D array
            .context("Failed to convert embedding to 2D array")?
            .to_owned(); // Clone the array for further use

        // Convert the first row of the 2D array into a Vec<f32>
        let embedding_vector: Vec<f32> = embedding.row(0).to_vec();

        println!("Extracted embedding with {} dimensions", embedding_vector.len());

        // Return the embedding vector
        Ok(embedding_vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[tokio::test]
    async fn test_embedding_extraction() {
        // Create a new instance of the embedding extractor
        let extractor = FishEmbeddingExtractor::new()
            .await
            .expect("Failed to create extractor");

        // Create a dummy input image (all zeros) with shape (3, 224, 224)
        let dummy_input = Array3::<f32>::zeros((3, 224, 224));

        // Extract the embedding from the dummy input
        let embedding = extractor
            .extract_embedding(dummy_input)
            .expect("Embedding extraction failed");

        // Print the dimension of the extracted embedding
        println!("Extracted embedding dimension: {}", embedding.len());

        // Ensure the embedding has 128 dimensions
        assert_eq!(embedding.len(), 128, "Embedding should have 128 dimensions");
    }
}