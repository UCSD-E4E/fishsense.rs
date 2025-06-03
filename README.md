# FishSense

## Introduction

FishSense is a Rust-based toolkit engineered for sophisticated analysis of fish imagery. It integrates machine learning models with advanced image processing algorithms to provide a comprehensive suite of functionalities. These range from identifying fish species and segmenting them from their aquatic environment, to precisely locating anatomical landmarks like the head and tail, and ultimately measuring their real-world length in 3D.

This document serves as an in-depth guide to the FishSense project. It meticulously explains the role and inner workings of each key file and module, clarifies the algorithms used, and details how these components interact to achieve FishSense's powerful capabilities. This README is designed for developers looking to understand, use, or contribute to the FishSense codebase.

## Core Functionalities at a Glance

* **Fish Species Classification**: Determines the species of fish present in an image.
* **Instance Segmentation**: Generates precise pixel-level masks to isolate fish from their background.
* **Automated Head/Tail Detection (Autolabeling)**: Pinpoints the 2D coordinates of a fish's head and tail on its segmentation mask using robust geometric and image analysis techniques.
* **3D Length Estimation**: Calculates the fish's actual length in three-dimensional space by combining 2D image points with depth information and camera parameters.

## Understanding Your FishSense Project: A Detailed File-by-File Exploration

This section provides a comprehensive breakdown of the FishSense project, detailing the purpose and technical aspects of each significant file and directory.

### Root Directory

* `Cargo.toml`
    * **Purpose**: The Rust project's manifest file, managing metadata, dependencies, and build configurations.
    * **Key Dependencies**:
        * `ort`: For running ONNX machine learning models (used for classification and segmentation).
        * `image`: For image loading, manipulation, and decoding/encoding.
        * `ndarray`, `ndarray-npy`, `ndarray-stats`: For numerical computations and N-dimensional arrays (tensors).
        * `opencv`, `cv-convert`: Provides access to the OpenCV computer vision library and utilities for data conversion.
        * `faer`: For high-performance linear algebra operations.
        * `reqwest`: For downloading files (e.g., models) over HTTP.
        * `serde`, `serde_json`: For serializing and deserializing data structures (e.g., JSON metadata).
        * `app_dirs2`: For finding appropriate user-specific cache directories (e.g., for downloaded models).
        * `anyhow`: For streamlined error handling.
    * *(Note: `autolabel.rs` also relies on `geo`, `imageproc`, and `nalgebra` for its geometric analysis, which should be listed here if not already present).*

* `rust-toolchain.toml`
    * **Purpose**: Specifies the exact Rust toolchain (compiler version and components) to ensure consistent and reproducible builds across different environments.

* `README.md`
    * **Purpose**: This document, providing an overview and guide to the FishSense Rust project.

### `src` Directory 

This directory contains all the Rust source code for the FishSense library.

* `src/main.rs`
    * **Purpose**: The entry point for a command-line executable that demonstrates fish classification.
    * **Functionality**:
        * Parses command-line arguments for an input image path.
        * Loads and preprocesses the image (resizing, normalization, tensor conversion) to match the classification model's input requirements.
        * Uses `FishClassifier` from the `fishsense` library to classify the image.
        * Prints the resulting species labels and confidence scores.

* `src/lib.rs`
    * **Purpose**: The main library crate file. It defines the public API of the FishSense library by declaring and re-exporting modules and their key components.
    * **Structure**: Declares public modules like `fish`, `world_point_handler`, and `linalg`, and may re-export primary structs (e.g., `FishClassifier`) for easier access by users of the library.

* `src/world_point_handler.rs`
    * **Purpose**: Defines the `WorldPointHandler` struct, responsible for converting 2D image coordinates to 3D real-world coordinates.
    * **Functionality**: Stores an inverted camera intrinsics matrix. Its main method takes 2D image coordinates and a depth value to compute the corresponding 3D point in world space.

* `src/linalg.rs`
    * **Purpose**: Provides utility functions for linear algebra operations, such as calculating the Euclidean norm (length) of a vector, used for distance computations between 3D points.

### `src/fish/` 

This module directory contains the specialized logic for analyzing fish.

* `src/fish/mod.rs`
    * **Purpose**: The root of the `fish` module. It declares its sub-modules (e.g., `fish_classifier`, `fish_segmentation`, `autolabel`, `fish_length_calculator`) and re-exports their primary public structs and functions.

* `src/fish/fish_classifier.rs`
    * **Purpose**: Implements `FishClassifier` for identifying fish species.
    * **Functionality**:
        * Manages downloading and caching of an ONNX classification model, a database of known fish embeddings (`embeddings.npy`), and associated metadata (`database.json`) from Hugging Face.
        * Initializes an ONNX session and loads the embedding database.
        * The `classify` method takes a preprocessed image tensor, runs it through the ONNX model to get an image embedding, and then finds the most similar embeddings in its database using cosine similarity to determine the fish species.

* `src/fish/fish_segmentation.rs`
    * **Purpose**: Implements `FishSegmentation` for generating pixel-accurate masks that isolate fish in images.
    * **Functionality**:
        * Downloads and caches an ONNX instance segmentation model (e.g., `fishial.onnx`).
        * Preprocesses input images (resizing, padding) for the model.
        * Runs inference to get bounding boxes, raw mask data, and scores.
        * Postprocesses the raw masks: filters by score, resizes masks to the original image scale, binarizes them, and extracts polygonal contours.
        * Includes test functions for validation.

* `src/fish/autolabel.rs`
    * **Purpose**: Implements `FishHeadTailDetector` to automatically identify head and tail points on a fish's segmentation mask.
    * **Functionality**:
        * Takes a grayscale fish mask as input.
        * Uses image analysis techniques, including Principal Component Analysis (PCA) on pixel coordinates, to determine the fish's primary orientation.
        * Extracts polygonal contours from the mask using `imageproc`.
        * Analyzes geometric properties of the contour and its convex hull using the `geo` library to distinguish and refine the head and tail points. This involves heuristics based on convexity and extremity along the fish's main axis.
        * Includes test functions with visual output for debugging.

* `src/fish/fish_length_calculator.rs`
    * **Purpose**: Implements `FishLengthCalculator` to determine the 3D physical length of a fish.
    * **Functionality**:
        * Requires a `WorldPointHandler`, image dimensions, a depth map (`Array2<f32>`), and the 2D image coordinates of the fish's head and tail.
        * Maps 2D image coordinates to depth map coordinates.
        * Implements a "depth snapping" logic (`snap_depth_coord`) to find reliable depth values on the fish's surface near the given 2D points, avoiding noisy background or edge values.
        * Converts the 2D head and tail points (with their snapped depth values) into 3D world coordinates using the `WorldPointHandler`.
        * Calculates the Euclidean distance between these two 3D points to get the fish's length.

### The `data` Directory

* **Purpose**: Contains sample images, NumPy arrays (`.npz`), and other data files used primarily by the test suite to validate the functionality of different modules.
* **Examples**:
    * `fish_segmentation.npz`: Used by `src/fish/fish_segmentation.rs` tests. Contains an image (`img8`) and its corresponding ground truth segmentation mask (`segmentations`).
    * `segmentations.png`: An example segmentation mask image, possibly used by older tests or `src/fish/autolabel.rs` tests.
    * `fish1.png` through `fish9.jpeg`, `test1.jpeg`, `test1_seg.jpeg`: Various raw fish images and pre-segmented masks used as input for tests in `src/fish/autolabel.rs` and `src/fish/fish_segmentation.rs`. Output images from tests, such as `fish1_out.png` (showing detected head/tail points), might also be saved into this directory by the test code itself for visual inspection.

### The `target` Directory

* **Purpose**: Automatically generated by Cargo (Rust's build system) to store all output from the compilation process.
* **Technical Details**: Contains subdirectories for debug and release builds (`debug/`, `release/`). Inside these, you'll find compiled library files (e.g., `.rlib`), executables, dependency build caches, and other intermediate artifacts. This directory is generally not manually modified or version-controlled (it's typically listed in `.gitignore`).

## Getting Started

### Prerequisites

* **Rust Programming Language**: Install Rust and its package manager, Cargo, via `rustup` from [rustup.rs](https://rustup.rs/). FishSense is built using the Rust 2021 edition and specifies a toolchain version in `rust-toolchain.toml`.
* **OpenCV Library**: Several image processing functionalities rely on OpenCV. You must have the OpenCV development libraries installed on your system. Detailed instructions for your operating system can be found in the `opencv-rust` crate's documentation: [opencv-rust Installation Guide](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md).
* **C++ Build Tools**: A C++ compiler (e.g., GCC on Linux, Clang on macOS, MSVC on Windows) is necessary for building certain Rust dependencies, particularly `ort` (ONNX Runtime), which has C++ core components.

### Installation

1.  **Obtain the Source Code**: If you have a Git repository URL:
    ```bash
    git clone <repository-url>
    cd fishsense
    ```
    If you have the source code as a directory, navigate into it.

2.  **Build the Project**:
    To compile FishSense for development and testing:
    ```bash
    cargo build
    ```
    For a production-ready, optimized build (especially for the command-line application or when using the library in a release context):
    ```bash
    cargo build --release
    ```

### Running the Command-Line Classifier

After a successful release build, you can execute the fish species classifier:
```bash
./target/release/fishsense /path/to/your/image.jpg

*Replace /path/to/your/image.jpg with the file path of the image you wish to classify. The application will output the predicted species names and their confidence scores.

