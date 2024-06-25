use ndarray::{array, Array2};

pub struct WorldPointHandler {
    pub camera_intrinsics_inverted: Array2<f32>
}

impl WorldPointHandler {
    pub fn compute_world_point_from_depth(&self, image_coordinate: &ndarray::Array1<f32>, depth: f32) -> ndarray::Array1<f32> {
        // The camera intrinsics includes the pixel pitch.
        self.camera_intrinsics_inverted.dot(&array![image_coordinate[0], image_coordinate[1], 1f32]) * depth
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::WorldPointHandler;

    #[test]
    fn compute_world_point_from_depth() {
        let image_point = array![889.63158192f32, 336.58548892f32];
        let depth = 0.5355310460918119f32;
        let camera_intrinsics_inverted = array![[0.00070161547, 0.0, 0.0], [0.0, 0.00070161547, 0.0], [-0.67513853, -0.5045314, 1.0]].t().mapv(|v| v as f32);

        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted
        };

        assert_eq!(world_point_handler.compute_world_point_from_depth(&image_point, depth), array![-0.02729025, -0.14372465, depth]);
    }
}