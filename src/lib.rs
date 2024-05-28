mod fish_segmentation;
mod fish_head_tail_detector;

pub mod fish {
    pub use crate::fish_segmentation::{FishSegmentation, SegmentationError};
    pub use crate::fish_head_tail_detector::FishHeadTailDetector;
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}