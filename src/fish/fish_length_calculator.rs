use ndarray::{array, Array1, Array2};

use crate::{linalg::norm, world_point_handlers::WorldPointHandler};

pub struct FishLengthCalculator<'world_point_handler> {
    pub world_point_handler: &'world_point_handler dyn WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize
}

impl <'world_point_handler> FishLengthCalculator<'world_point_handler> {
    fn get_depth(&self, depth_mask: &Array2<f32>, img_coord: &Array1<f32>) -> f32 {
        println!("RUST: Start Get Coord Depth");
        let (height, width) = depth_mask.dim();

        let height_f32 = height as f32;
        let width_f32 = width as f32;
        let img_height_f32 = self.image_height as f32;
        let img_width_f32 = self.image_width as f32;
        let coord_f32 = img_coord / array![img_height_f32, img_width_f32] * array![height_f32, width_f32];

        let coord = coord_f32.mapv(|v| v as usize);

        println!("RUST: height, width: ({}, {}); coord: ({})", height, width, coord);
        println!("RUST: img_height, img_width: ({}, {}); img_coord: ({})", self.image_height, self.image_width, img_coord);
        let result = depth_mask[[coord[0], coord[1]]];

        println!("RUST: End Get Coord Depth");

        result
    }

    pub fn calculate_fish_length(&self, depth_mask: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> f32 {
        println!("RUST: Start Calculate Fish Length");

        let left_depth = self.get_depth(depth_mask, left_img_coord);
        let right_depth = self.get_depth(depth_mask, right_img_coord);

        println!("RUST: ({}, {})", left_depth, right_depth);

        println!("RUST: Start Calculate world point handler");
        let left_3d = self.world_point_handler.compute_world_point_from_depth(&left_img_coord, left_depth);
        let right_3d = self.world_point_handler.compute_world_point_from_depth(&right_img_coord, right_depth);
        println!("RUST: End Calculate world point handler");

        println!("RUST: left_3d: {}, right_3d: {}", left_3d, right_3d);

        let result = norm(&(left_3d - right_3d));

        println!("RUST: End Calculate Fish Length");

        result
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::world_point_handlers::PixelPitchWorldPointHandler;

    use super::FishLengthCalculator;

    #[test]
    fn calculate_fish_length() {
        let depth_mask = array![[0.5355310460918119f32]];

        let pixel_pitch_mm = 0.0015;
        let focal_length_mm = 4.247963447392709;
        let world_point_handler = PixelPitchWorldPointHandler {
            focal_length_mm,
            pixel_pitch_mm
        };

        let image_height = 3016;
        let image_width = 3987;
        let fish_length_calcualtor = FishLengthCalculator {
            image_height,
            image_width,
            world_point_handler: &world_point_handler
        };

        let left = array![889.63158192f32, 336.58548892f32];
        let right = array![-355.36841808f32, 395.58548892f32];
        let fish_length = fish_length_calcualtor.calculate_fish_length(&depth_mask, &left, &right);
        
        assert_eq!(fish_length, 0.23569568f32);
    }
}