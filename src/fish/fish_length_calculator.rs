use ndarray::{array, Array1, Array2};
use num::complex::ComplexFloat;

use crate::{linalg::norm, WorldPointHandler};

pub struct FishLengthCalculator {
    pub world_point_handler: WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize
}

impl FishLengthCalculator {
    fn get_depth(&self, depth_map: &Array2<f32>, depth_coord: &Array1<f32>) -> f32 {
        let depth_coord = depth_coord.mapv(|v| v as usize);

        depth_map[[depth_coord[0], depth_coord[1]]]
    }

    fn get_depth_coord(&self, depth_map: &Array2<f32>, img_coord: &Array1<f32>) -> Array1<f32> {
        let (height, width) = depth_map.dim();

        let height_f32 = height as f32;
        let width_f32 = width as f32;
        let img_height_f32 = self.image_height as f32;
        let img_width_f32 = self.image_width as f32;

        img_coord / array![img_height_f32, img_width_f32] * array![height_f32, width_f32]
    }

    fn snap_depth_coord(&self, depth_map: &Array2<f32>, depth_coord: &Array1<f32>, mid_coord: &Array1<f32>) -> Array1<usize> {
        let (height, width) = depth_map.dim();
        let height = height as f32;
        let width = width as f32;

        let dir = (mid_coord - depth_coord) / norm(&(mid_coord - depth_coord));
        let mut coord = mid_coord.clone();
        let mut depth = self.get_depth(depth_map, &coord);
        let mut previous_depth = depth.clone();

        let mut failed = false;
        while (depth - previous_depth).abs() < 0.005 {
            previous_depth = depth.clone();
            
            coord += &dir;

            if coord[0] <= 0f32 || coord[0] >= height || coord[1] <= 0f32 || coord[1] >= width {
                failed = true;
            }

            depth = self.get_depth(depth_map, &coord);
        }
        coord -= &dir;

        if failed {
            coord = depth_coord.clone();
        }

        coord.mapv(|v| v as usize)
    }

    fn get_depths(&self, depth_map: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> (f32, f32) {
        let left_coord_f32 = self.get_depth_coord(depth_map, left_img_coord);
        let right_coord_f32 = self.get_depth_coord(depth_map, right_img_coord);

        let mid_coord_f32 = &left_coord_f32 + (&right_coord_f32 - &left_coord_f32) / 2f32;

        let left_coord = self.snap_depth_coord(depth_map, &left_coord_f32, &mid_coord_f32);
        let right_coord = self.snap_depth_coord(depth_map, &right_coord_f32, &mid_coord_f32);

        let mid_coord = mid_coord_f32.mapv(|v| v as usize);

        let left_depth = depth_map[[left_coord[0], left_coord[1]]];
        let right_depth = depth_map[[right_coord[0], right_coord[1]]];

        let mid_depth = depth_map[[mid_coord[0], mid_coord[1]]];

        println!("RUST: {}, {}, {}", left_depth, right_depth, mid_depth);

        (depth_map[[left_coord[0], left_coord[1]]], depth_map[[right_coord[0], right_coord[1]]])
    }

    pub fn calculate_fish_length(&self, depth_map: &Array2<f32>, left_img_coord: &Array1<f32>, right_img_coord: &Array1<f32>) -> f32 {
        let (left_depth, right_depth) = self.get_depths(depth_map, left_img_coord, right_img_coord);

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
        let depth_map = array![[0.5355310460918119f32]];
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
        let fish_length = fish_length_calcualtor.calculate_fish_length(&depth_map, &left, &right);
        
        assert_eq!(fish_length, 0.23569532);
    }
}