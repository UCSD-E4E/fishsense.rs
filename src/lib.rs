mod fish_segmentation;
mod fish_head_tail_detector;

pub use fish_segmentation::FishSegmentation;
pub use fish_head_tail_detector::FishHeadTailDetector;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}