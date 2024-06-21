use ndarray::{array, Array1, Array2};

use crate::{linalg::norm, WorldPointHandler};

pub struct FishLengthCalculator {
    pub world_point_handler: WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize
}

impl FishLengthCalculator {
    fn get_depth_coord(&self, depth_mask: &Array2<f32>, img_coord: &Array1<f32>) -> Array1<usize> {
        let (height, width) = depth_mask.dim();

        let height_f32 = height as f32;
        let width_f32 = width as f32;
        let img_height_f32 = self.image_height as f32;
        let img_width_f32 = self.image_width as f32;
        let coord_f32 = img_coord / array![img_height_f32, img_width_f32] * array![height_f32, width_f32];

        coord_f32.mapv(|v| v as usize)
    }

    fn get_depths(&self, depth_mask: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> (f32, f32) {
        let left_coord = self.get_depth_coord(depth_mask, left_img_coord);
        let right_coord = self.get_depth_coord(depth_mask, right_img_coord);

        let mut left_direction = (&right_coord - &left_coord).mapv(|v| v as f32);
        left_direction /= norm(&left_direction);

        let mut right_direction = (&left_coord - &right_coord).mapv(|v| v as f32);
        right_direction /= norm(&right_direction);

        let left_step = (left_coord.mapv(|v| v as f32) + &left_direction / array![left_direction[0], left_direction[1]]).mapv(|v| v as usize);
        let right_step = (right_coord.mapv(|v| v as f32) + &right_direction / array![right_direction[0], right_direction[1]]).mapv(|v| v as usize);

        println!("RUST: {} {} left depth: {}, step depth: {}", left_coord, left_step, depth_mask[[left_coord[0], left_coord[1]]], depth_mask[[left_step[0], left_step[1]]]);
        println!("RUST: {} {} right depth: {}, step depth: {}", right_coord, right_step, depth_mask[[right_coord[0], right_coord[1]]], depth_mask[[right_step[0], right_step[1]]]);

        (depth_mask[[left_coord[0], left_coord[1]]], depth_mask[[right_coord[0], right_coord[1]]])
    }

    pub fn calculate_fish_length(&self, depth_mask: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> f32 {
        let (left_depth, right_depth) = self.get_depths(depth_mask, left_img_coord, right_img_coord);

        let left_3d = self.world_point_handler.compute_world_point_from_depth(&left_img_coord, left_depth);
        let right_3d = self.world_point_handler.compute_world_point_from_depth(&right_img_coord, right_depth);

        norm(&(left_3d - right_3d))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::WorldPointHandler;

    use super::FishLengthCalculator;

    #[test]
    fn calculate_fish_length() {
        let depth_mask = array![[0.5355310460918119f32]];
        let f_inv = 0.00035310983834631600505f32;
        let camera_intrinsics_inverted = array![[f_inv, 0f32, 0f32], [0f32, f_inv, 0f32], [0f32, 0f32, 1f32]];

        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted
        };

        let image_height = 3016;
        let image_width = 3987;
        let fish_length_calcualtor = FishLengthCalculator {
            image_height,
            image_width,
            world_point_handler
        };

        let left = array![889.63158192f32, 336.58548892f32];
        let right = array![-355.36841808f32, 395.58548892f32];
        let fish_length = fish_length_calcualtor.calculate_fish_length(&depth_mask, &left, &right);
        
        assert_eq!(fish_length, 0.23569532);
    }
}