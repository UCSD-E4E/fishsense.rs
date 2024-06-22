use ndarray::{array, Array1, Array2};

use crate::{linalg::norm, WorldPointHandler};

pub struct FishLengthCalculator {
    pub world_point_handler: WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize
}

impl FishLengthCalculator {
    fn get_depth_at_coord(&self, depth_mask: &Array2<f32>, coord: &Array1<f32>) -> f32 {
        let coord_usize = coord.map(|v| v.to_owned() as usize);

        depth_mask[[coord_usize[0], coord_usize[1]]]
    }

    fn walk(&self, depth_mask: &Array2<f32>, mid_point: &Array1<f32>, direction: &Array1<f32>, stopping_point: &Array1<f32>) -> Array1<usize> {
        let mut coord = mid_point.clone();
        let mut depth = self.get_depth_at_coord(depth_mask, &coord);
        let mut prev_depth = depth.clone();

        while (depth - prev_depth).abs() < 0.001 {
            prev_depth = depth;
            coord += direction;

            if coord[0] >= stopping_point[0] || coord[1] >= stopping_point[1] {
                coord = stopping_point.clone();
                break
            }

            depth = self.get_depth_at_coord(depth_mask, &coord);
        }
        println!("RUST: depth: {}, prev_depth: {}", depth, prev_depth);

        coord.mapv(|v| v as usize)
    }

    fn get_depth_coord(&self, depth_mask: &Array2<f32>, img_coord: &Array1<f32>) -> Array1<f32> {
        let (height, width) = depth_mask.dim();

        let height_f32 = height as f32;
        let width_f32 = width as f32;
        let img_height_f32 = self.image_height as f32;
        let img_width_f32 = self.image_width as f32;

        img_coord / array![img_height_f32, img_width_f32] * array![height_f32, width_f32]
    }

    fn get_depths(&self, depth_mask: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> (f32, f32) {
        let left_coord = self.get_depth_coord(depth_mask, left_img_coord);
        let right_coord = self.get_depth_coord(depth_mask, right_img_coord);

        let mid_point = &right_coord + (&right_coord - &left_coord) / 2f32;

        let mut left_direction = &right_coord - &left_coord;
        left_direction /= norm(&left_direction);

        let mut right_direction = &right_coord - &left_coord;
        right_direction /= norm(&right_direction);

        let left_coord = self.walk(depth_mask, &mid_point, &left_direction, &left_coord);
        let right_coord = self.walk(depth_mask, &mid_point, &right_direction, &right_coord);

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