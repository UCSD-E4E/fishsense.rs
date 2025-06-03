# FishSense

**FishSense** is a Rust toolkit for analyzing fish images using machine learning and image processing. It classifies species, segments fish from backgrounds, detects head/tail points, and estimates 3D length using depth data.

---

## Features

- **Species Classification**: Identifies fish species via ONNX models.
- **Segmentation**: Extracts precise fish masks from images.
- **Head/Tail Detection**: Locates anatomical points using geometry and PCA.
- **3D Length Estimation**: Computes real-world fish length using image coordinates, depth maps, and camera intrinsics.

---

## Project Structure

### Root

- `Cargo.toml`: Rust manifest. Key crates: `ort`, `opencv`, `image`, `ndarray`, `faer`, `serde`, `anyhow`, `reqwest`.
- `rust-toolchain.toml`: Pins the Rust version.
- `README.md`: You're reading it.

### `src/`

- `main.rs`: CLI for running classification on a fish image.
- `lib.rs`: Exposes library modules.

#### Core Modules

- `world_point_handler.rs`: Converts 2D points + depth → 3D world coordinates.
- `linalg.rs`: Vector math utilities.

#### `fish/` Module

- `mod.rs`: Declares and re-exports fish analysis modules.
- `fish_classifier.rs`: Loads ONNX model + embedding DB; runs cosine similarity to identify species.
- `fish_segmentation.rs`: Runs ONNX instance segmentation model; outputs binary masks and contours.
- `autolabel.rs`: Detects head/tail points using PCA and convex hull analysis.
- `fish_length_calculator.rs`: Snaps head/tail to best depth values; computes 3D length.

---

## `data/` Directory

Test files: sample fish images, segmentation masks, `.npz` arrays, etc. Used in module tests. Output images may also be saved here.

---

## `target/` Directory

Build artifacts—ignored by Git. Created by Cargo during compilation.

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
