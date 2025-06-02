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

### Root Directory Files

These essential files are located at the top level of your FishSense project.

* `Cargo.toml`
    * **Purpose**: The manifest file for the Rust project, acting as its central configuration.
    * **Technical Details**:
        * Defines project metadata: `name` ("fishsense"), `version`, `edition` (Rust 2021).
        * Lists external dependencies (crates):
            * `ort`: Crucial for running inference with ONNX (Open Neural Network Exchange) models, which FishSense uses for both classification and segmentation. This specific version is pulled from a Git repository, possibly for features not yet in a crates.io release.
            * `image`: A general-purpose image encoding and decoding library, used for loading and manipulating image files.
            * `ndarray`, `ndarray-npy`, `ndarray-stats`: Fundamental for numerical computing. `ndarray` provides N-dimensional array (tensor) capabilities. `ndarray-npy` allows reading NumPy's `.npy` and `.npz` file formats (useful for model embeddings or test data). `ndarray-stats` (from a specific Git source) adds statistical operations on these arrays.
            * `opencv` & `cv-convert`: Provide bindings to the OpenCV library, a comprehensive suite of computer vision algorithms. `cv-convert` likely facilitates conversions between `ndarray` structures and OpenCV's `Mat` objects.
            * `faer`: A high-performance Rust library for linear algebra, potentially used for matrix decompositions or other complex calculations.
            * `reqwest`: An HTTP client used for downloading files from the internet, such as ONNX models or metadata, with the "blocking" feature enabled for synchronous requests.
            * `serde` & `serde_json`: For serializing Rust data structures into formats like JSON and deserializing them back, particularly used for configuration or metadata files.
            * `app_dirs2`: Helps in finding standard user-specific cache directories to store downloaded models persistently.
            * `anyhow`: A library for flexible and user-friendly error handling, simplifying error propagation.
        * **Note for `autolabel.rs`**: To use the `autolabel.rs` module with its full capabilities (geometric analysis, contour finding), you will likely need to add `geo`, `imageproc`, and `nalgebra` to this `[dependencies]` section.

* `rust-toolchain.toml`
    * **Purpose**: Ensures build consistency by specifying the exact Rust toolchain.
    * **Technical Details**: It dictates the Rust compiler version (e.g., `channel = "1.84"`) and components to be used. This is vital for reproducible builds across different development environments and build systems, preventing issues caused by toolchain discrepancies.

* `README.md`
    * **Purpose**: This document – providing comprehensive information about the FishSense project.
    * **Technical Details**: It's written in Markdown and serves as the primary guide for users and developers.

### The `src` Directory (Source Code)

This directory houses all the Rust source code that constitutes the FishSense library and application.

* `src/main.rs`
    * **Purpose**: The entry point for the FishSense command-line executable, specifically for fish classification.
    * **Technical Details**:
        1.  **Argument Parsing**: Reads command-line arguments to get the path to an input image.
        2.  **Image Loading & Preprocessing**:
            * Uses the `image` crate to open the specified image file.
            * Resizes the image to a fixed size (e.g., 224x224 pixels), as expected by the classification model.
            * Converts the image to RGB8 format.
            * Transforms the pixel data into an `ndarray::Array3<f32>` tensor. This involves:
                * Changing pixel order if necessary (e.g., from RGB to BGR if the model expects it, though the example shows BGR-like assignment: `pixel[2]` (Blue), `pixel[1]` (Green), `pixel[0]` (Red) mapped to channels 0, 1, 2 respectively).
                * Normalizing pixel values (e.g., dividing by 255.0 to scale them to the [0.0, 1.0] range).
                * Arranging data in CHW (Channels, Height, Width) format if that's what the ONNX model's input layer requires.
        3.  **Classification**: Initializes a `FishClassifier` instance (from the `fishsense` library) and calls its `classify` method with the preprocessed image tensor.
        4.  **Output**: Prints the classification results (species labels and confidence scores) to the console.

* `src/lib.rs`
    * **Purpose**: The main entry point and public API definition for the FishSense library.
    * **Technical Details**:
        * It declares the top-level public modules of the library using `pub mod <module_name>;` (e.g., `pub mod fish;`, `pub mod world_point_handler;`).
        * It may also re-export important structs, enums, or functions from these modules to create a more convenient and consolidated public interface for library users. For example, `pub use fish::FishClassifier;`.

* `src/world_point_handler.rs` (Assuming this file path based on common Rust project structure)
    * **Purpose**: Defines the `WorldPointHandler` struct and its logic for converting 2D image coordinates into 3D real-world coordinates.
    * **Technical Details**:
        * The `WorldPointHandler` struct stores `camera_intrinsics_inverted` (an `Array2<f32>`). The camera intrinsics matrix describes the internal geometric properties of a camera (focal length, principal point, pixel size). The *inverted* intrinsics matrix is used to project points from the 2D image plane back into 3D space.
        * The core method `compute_world_point_from_depth(&self, image_coordinate: &ndarray::Array1<f32>, depth: f32) -> ndarray::Array1<f32>` implements this projection. Conceptually, if `K_inv` is the inverted intrinsics matrix, `[u, v]` are the image coordinates, and `Z` is the depth, the 3D point `[X, Y, Z]` is found by `[X', Y', W']^T = K_inv * [u, v, 1]^T`, and then `X = X'/W' * Z`, `Y = Y'/W' * Z`. The provided code simplifies this: `K_inv.dot(&array![image_coordinate[0], image_coordinate[1], 1f32]) * depth`, assuming the intrinsics matrix correctly incorporates scaling to world units.

* `src/linalg.rs` (Assumed file path based on `use crate::linalg::norm;` in `fish_length_calculator.rs`)
    * **Purpose**: This file likely provides custom or helper linear algebra functions not directly available or conveniently packaged by the main dependencies.
    * **Technical Details**: Based on its usage in `FishLengthCalculator` (`crate::linalg::norm`), it defines at least a `norm` function. This function would calculate the Euclidean norm (or magnitude/length) of a vector, typically as the square root of the sum of the squares of its components. This is fundamental for calculating distances between 3D points. While `ndarray` itself has norm calculation methods, this custom function might be tailored for specific types or error handling within the project.

### The `src/fish/` Module Directory: Core Fish Analysis Logic

This directory encapsulates the specialized modules for detailed fish analysis.

* `src/fish/mod.rs`
    * **Purpose**: Acts as the root of the `fish` module, organizing its sub-components.
    * **Technical Details**:
        * It declares all sub-modules within the `fish` directory using `mod <sub_module_name>;` (e.g., `mod fish_classifier;`, `mod autolabel;`).
        * It then uses `pub use <sub_module_name>::{StructName, EnumName};` statements to selectively re-export the primary public interfaces from these sub-modules. This allows users of the `fishsense::fish` module to access these items directly (e.g., `fishsense::fish::FishClassifier`) without needing to know the internal file structure.

* `src/fish/fish_classifier.rs`
    * **Purpose**: Implements the `FishClassifier` for identifying fish species.
    * **Technical Details**:
        * **Data Management**: Downloads and caches three key files from Hugging Face:
            1.  `fish_classifier.onnx`: The pre-trained ONNX deep learning model that extracts feature embeddings from images.
            2.  `embeddings.npy`: A NumPy array file containing a database of pre-computed feature embeddings for known fish images/species.
            3.  `database.json`: Metadata linking the embeddings in `embeddings.npy` to human-readable labels (e.g., fish species names) and other information. Contains a custom deserializer (`deserialize_keys_int_to_string`) to handle potentially numeric keys in the JSON being treated as strings.
        * **Initialization (`FishClassifier::new()`)**:
            * Ensures the above files are downloaded to a user cache directory (via `app_dirs2`).
            * Loads the ONNX model into an `ort::Session`.
            * Reads the `embeddings.npy` into an `ndarray::Array2<f32>`.
            * Parses `database.json` into the `Metadata` struct.
        * **Classification (`classify(&self, input_tensor: Array3<f32))`)**:
            1.  Takes a preprocessed input image tensor (`Array3<f32>`, typically CHW format, normalized).
            2.  Runs this tensor through the ONNX model. The model outputs include:
                * An `embedding` vector (`Array2<f32>` of shape [1, embedding_size]) for the input image, representing its features.
                * `logits` (`Array2<f32>` of shape [1, num_classes]), which are raw scores for each class the base model might have been trained on.
            3.  **Similarity Search**: Calculates the cosine similarity between the input image's `embedding` and every embedding in the loaded `self.embeddings` database. Cosine similarity measures the cosine of the angle between two vectors, indicating how similar their directions are (closer to 1 means more similar).
            4.  **Top-K Results**: Identifies the `k` database embeddings most similar to the input embedding.
            5.  **Label Retrieval**: Uses `self.metadata` to map the indices of these top-matching database embeddings back to their corresponding fish species labels.
            6.  Returns a `Vec<(String, f32)>` of (label, similarity_score) pairs.

* `src/fish/fish_segmentation.rs`
    * **Purpose**: Implements `FishSegmentation` for creating pixel-accurate masks that isolate fish in images.
    * **Technical Details**:
        * **Model Handling**: Similar to the classifier, it downloads and caches an ONNX model (`fishial.onnx` from Hugging Face), specifically one trained for instance segmentation tasks.
        * **Image Preprocessing (`resize_img`, `pad_img`)**:
            * `resize_img`: Resizes the input image (`Array3<u8>`) to dimensions suitable for the ONNX model (e.g., `MIN_SIZE_TEST`, `MAX_SIZE_TEST`), maintaining aspect ratio up to a point. It uses OpenCV via `cv-convert` for resizing.
            * `pad_img`: If the resized image doesn't exactly match the model's required input dimensions (e.g., after aspect-ratio-preserving resize), it pads the image with zeros to fit.
        * **Inference (`do_inference`)**: The preprocessed image (converted to `f32`) is fed into the ONNX model. The model is expected to output:
            * `boxes`: Bounding box coordinates for detected instances.
            * `masks`: Raw mask data for each instance, often lower resolution than the input image and requiring further processing.
            * `scores`: Confidence scores for each detected instance.
        * **Mask Postprocessing (`do_paste_mask`, `bitmap_to_polygon`, `convert_output_to_mask_and_polygons`)**:
            1.  Filters detections based on `SCORE_THRESHOLD`.
            2.  For each valid detection, its raw mask data is resized (`do_paste_mask` using OpenCV's `resize_def`) to match the original image's region of interest (defined by the bounding box).
            3.  The resized mask is binarized using `MASK_THRESHOLD`.
            4.  Contours (polygons) are extracted from this binary mask (`bitmap_to_polygon` using OpenCV's `find_contours_with_hierarchy`). This helps in getting clean outlines.
            5.  These polygons are rescaled to the original image dimensions and potentially drawn onto a final output mask (`Array2<u8>`) for the entire image, with different instances possibly getting different small integer labels.
        * **Error Handling**: Defines a `SegmentationError` enum to cover various issues like download errors, model loading problems, OpenCV errors, etc.
        * **Tests**: Includes an `inference` test using `data/fish_segmentation.npz` (which contains an image and ground truth segmentation) and an `inference1` test processing a JPEG image and saving the output segmentation mask.

* `src/fish/autolabel.rs`
    * **Purpose**: Contains the sophisticated `FishHeadTailDetector` for automatically identifying head and tail points on a fish, using a grayscale image mask as input.
    * **Technical Details (`find_head_tail` method)**:
        1.  **Input Conversion**: Converts the input `ImageBuffer<Luma<u8>, Vec<u8>>` (grayscale image mask) into an `ndarray::Array2<u8>`.
        2.  **Non-Zero Pixel Extraction & Bounding Box**: Identifies all non-zero (fish) pixels. Calculates the bounding box (`x_min`, `y_min`, `x_max`, `y_max`) around these pixels and crops the mask to this region to focus computation.
        3.  **Coordinate Centering & Covariance**: Extracts coordinates of fish pixels within the crop. Centers these coordinates by subtracting their respective means (`x_mean`, `y_mean`). Then, it computes the 2x2 covariance matrix of these centered (x, y) coordinates (`compute_covariance` function using `nalgebra::Matrix2`). The covariance matrix describes the variance and correlation of the x and y pixel distributions.
        4.  **Principal Component Analysis (PCA)**: Performs symmetric eigendecomposition on the covariance matrix (`covariance_matrix.symmetric_eigen()`). The eigenvector corresponding to the largest eigenvalue (the "principal eigenvector") indicates the primary axis of variance in the pixel distribution, which usually aligns with the fish's longest dimension (body orientation).
        5.  **Line Projection & Initial Points**: Defines a line along this principal axis. Projects points far out along this line in both directions to get initial, rough estimates for the extremities.
        6.  **Refined Point Selection on Axis**: Narrows down candidates by finding points on the fish's body that lie closest to this principal axis within the cropped region. The furthest of these points along the axis become the new `left_coord` and `right_coord`.
        7.  **Image Resizing for Polygon Extraction**: The cropped image part of the original mask is further resized (e.g., to `TARGET_PIXELS`) using Lanczos3 resampling. This step likely aims to standardize the scale for polygon analysis or reduce computational load while preserving features.
        8.  **Polygon Extraction (`extract_polygon`)**: Uses `imageproc::contours::find_contours_with_threshold` on the resized grayscale mask to find pixel contours. The largest contour is selected and converted into a `geo::Polygon<f64>`.
        9.  **Convex Hull**: Computes the convex hull of this polygon (`poly.convex_hull()`). The convex hull is the smallest convex shape enclosing all points of the polygon.
        10. **Head/Tail Disambiguation (`tail_head_distinct`)**: Distinguishes the more pointed/tapered end (head) from the potentially broader or more concave end (tail). This implementation divides the fish pixels into two halves based on a line perpendicular to the main axis and compares the "convexity loss" (area difference between a shape and its convex hull) of these halves. The half with greater convexity loss (more凹陷) is often associated with the tail region.
        11. **Tail Correction (`tail_correct`)**: Refines the tail point. It searches for the point on the *original polygon* (not the hull) that is maximally distant from the *convex hull* within a search radius around the current tail estimate. This helps identify points in tail fin concavities.
        12. **Head Correction (`head_correct`)**: Refines the head point. It searches for the point on the *convex hull* that is furthest along the fish's principal direction (vector from refined tail to current head) when considering points near a line perpendicular to this direction at the head. This tends to find the most extreme point of the head's smooth outline.
        13. **Coordinate Transformation**: All refined coordinates (initially in the resized, cropped space) are transformed back to the original image's coordinate system.
        14. **Visualization (`draw_dot`)**: A helper function is used to draw dots on the input `ImageBuffer` at various stages (initial points, final head/tail), useful for debugging and visual confirmation in tests. The tests in this module (`test_fish1` to `test_fish10`) save these modified images.
        * **Key Libraries**: `nalgebra` for PCA-like math, `imageproc` for contour detection, `geo` for polygon operations and convex hulls.

* `src/fish/fish_length_calculator.rs`
    * **Purpose**: Implements the `FishLengthCalculator` struct to determine the 3D physical length of a fish.
    * **Technical Details**:
        1.  **Inputs**: Requires a `WorldPointHandler` instance, the original image's height and width (to relate to depth map dimensions), a depth map (`Array2<f32>`), and the 2D image coordinates of the fish's left and right extremities (e.g., head and tail points from `autolabel.rs`).
        2.  **Depth Coordinate Mapping (`get_depth_coord`)**: Translates the input 2D image coordinates (which are in the original image's pixel space) to corresponding coordinates in the depth map's pixel space, if their dimensions differ.
        3.  **Depth Snapping (`snap_depth_coord`)**: This is a crucial step for robust depth retrieval. Given an initial 2D coordinate in the depth map and a direction (e.g., towards the fish's midpoint), it iteratively steps along this direction. It looks for a significant change in depth values, aiming to find a point on the fish's surface rather than potentially noisy background or edge depth values. If it goes out of bounds or fails to find a stable point, it may revert to the initial coordinate. This helps ensure the depth value used corresponds to the fish itself.
        4.  **3D Point Conversion**: Uses `get_depths` (which calls `snap_depth_coord`) to get reliable depth values for both the left and right 2D points. Then, it calls `world_point_handler.compute_world_point_from_depth` for each point to get their 3D coordinates.
        5.  **Length Calculation**: Computes the Euclidean distance between the two resulting 3D points using a `norm` function (likely from `crate::linalg` or `ndarray`). This distance is the calculated fish length.

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

