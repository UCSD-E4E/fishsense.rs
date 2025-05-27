mod fish_segmentation;
mod fish_length_calculator;
mod fish_classifier;
mod autolabel;


pub use fish_segmentation::{FishSegmentation, SegmentationError};
pub use fish_length_calculator::FishLengthCalculator;
pub use fish_classifier:: {FishClassifier};
pub use autolabel::{FishHeadTailDetector, HeadTailError};