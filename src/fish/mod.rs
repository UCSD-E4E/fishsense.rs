mod fish_segmentation;
// mod fish_head_tail_detector;
mod fish_length_calculator;
mod autolabel;


pub use fish_segmentation::{FishSegmentation, SegmentationError};
// pub use autolabel_old::{FishHeadTailDetector, HeadTailError};
pub use fish_length_calculator::FishLengthCalculator;
pub use autolabel::{FishHeadTailDetector, HeadTailError};

