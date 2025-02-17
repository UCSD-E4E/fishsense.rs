mod fish_segmentation;
mod fish_head_tail_detector;
mod fish_length_calculator;
mod fish_tail_correct;


pub use fish_segmentation::{FishSegmentation, SegmentationError};
// pub use fish_head_tail_detector::{FishHeadTailDetector, HeadTailError};
pub use fish_length_calculator::FishLengthCalculator;
pub use fish_tail_correct::{FishHeadTailDetector, HeadTailError};

