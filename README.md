# FishSense

**FishSense** is a Rust toolkit for analyzing fish images using machine learning and image processing. It classifies species, segments fish from backgrounds, detects head/tail points, and estimates fish length using depth data. The system features a vector database architecture optimized for edge deployment and real-time classification.

---

## Features

- **Species Classification**: Identifies fish species via ONNX models.
- **Segmentation**: Extracts precise fish masks from images.
- **Head/Tail Detection**: Locates points using fish geometry and PCA.
- **3D Length Estimation**: Computes real-world fish length using image coordinates, depth maps, and camera intrinsics.
- **Vector Database**: Similarity search across 69,990+ species embeddings using LanceDB.

---

## Project Structure

### Root

- `Cargo.toml`: Rust manifest. Key crates: `ort`, `opencv`, `image`, `ndarray`, `faer`, `serde`, `anyhow`, `reqwest`, `lancedb`, `arrow-array`, `tokio`. 
- `rust-toolchain.toml`: Pins the Rust version.

### `src/`

- `main_old.rs`: Original manual CLI for running classification on a fish image.
- `main.rs`: Vector database-powered CLI for fish classification on a fish image.
- `lib.rs`: Exposes library modules.
- `migrate.rs`: Automated migration system for embedding database setup.

#### Core Modules

- `world_point_handler.rs`: Converts 2D points + depth → 3D world coordinates.
- `linalg.rs`: Vector math utilities.

#### `fish/` Module

- `mod.rs`: Declares and re-exports fish analysis modules.
- `fish_classifier.rs`: Loads ONNX model + embedding DB; runs cosine similarity to identify species.
- `fish_segmentation.rs`: Runs ONNX instance segmentation model; outputs binary masks and contours.
- `autolabel.rs`: Detects head/tail points using convex hull analysis.
- `fish_length_calculator.rs`: Snaps head/tail to best depth values; computes length.
- `vector_db_classifier.rs`: LanceDB-powered classifier with normalized embeddings and cosine similarity.

---

## `data/` Directory

fish_lancedb/: Vector database containing normalized embeddings and species metadata.
Test files: sample fish images, segmentation masks, `.npz` arrays, etc. Used in module tests. 

---

## `target/` Directory

Created by Cargo during compilation.

---

## Getting Started

### Requirements

- **Rust** (via [rustup.rs](https://rustup.rs/))
- **OpenCV** development libraries: [opencv-rust install guide](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)
- **C++ toolchain**: Required by `ort` for ONNX Runtime.

### Build

```bash
git clone <repo-url>
cd fishsense
cargo build             # dev build
cargo build --release   # optimized build



```

### Usage

```bash
- On first execution, the system will automatically:

- Download the ONNX model and embedding data
- Create the vector database with normalized embeddings
- Build search indexes 


```

## More Details
https://anushri1206.github.io/FishSenseDoc/

## Acknowledgment

This documentation was partially drafted and edited with the assistance of AI-based tools, including [OpenAI ChatGPT](https://openai.com/chatgpt), to enhance clarity, consistency, and formatting.

