use ndarray::{array, Array1};

use crate::linalg::norm;

pub struct WorldPointHandler {
    pub focal_length_mm: f32,
    pub pixel_pitch_mm: f32
}

impl WorldPointHandler {
    pub fn image_coordinate_to_projected_point(&self, image_coordinate: &Array1<f32>) -> Array1<f32> {
        let projected_point = image_coordinate * self.pixel_pitch_mm / 1e3;
        return array![projected_point[0], projected_point[1], -self.focal_length_mm / 1e3];
    }

    pub fn compute_world_point_from_depth(&self, image_coordinate: &Array1<f32>, depth: f32) -> Array1<f32> {
        let projected_point = self.image_coordinate_to_projected_point(image_coordinate);
        let v = -&projected_point / norm(&projected_point);

        return &v * depth / v[2];
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::WorldPointHandler;

    #[test]
    fn image_coordinate_to_projected_point() {
        let image_point = array![128.63158192f32, 401.58548892f32];
        let pixel_pitch_mm = 0.0015;
        let focal_length_mm = 4.247963447392709;

        let world_point_handler = WorldPointHandler {
            focal_length_mm,
            pixel_pitch_mm
        };

        assert_eq!(world_point_handler.image_coordinate_to_projected_point(&image_point), array![0.00019294737f32, 0.0006023783f32, -0.0042479634f32]);
    }

    #[test]
    fn compute_world_point_from_depth() {
        let image_point = array![889.63158192f32, 336.58548892f32];
        let depth = 0.5355310460918119f32;
        let pixel_pitch_mm = 0.0015;
        let focal_length_mm = 4.247963447392709;

        let world_point_handler = WorldPointHandler {
            focal_length_mm,
            pixel_pitch_mm
        };

        assert_eq!(world_point_handler.compute_world_point_from_depth(&image_point, depth), array![-0.16823073, -0.06364884, 0.53553105]);
    }
}