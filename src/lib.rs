mod fish_segmentation;
mod fish_head_tail_detector;

pub use fish_segmentation::{FishSegmentation, SegmentationError};
pub use fish_head_tail_detector::FishHeadTailDetector;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}