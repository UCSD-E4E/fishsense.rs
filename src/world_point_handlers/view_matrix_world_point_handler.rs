use ndarray::{array, s, Array1, Array2};

use super::WorldPointHandler;

pub struct ViewMatrixWorldPointHandler {
    pub camera_intrinsics_inverted: Array2<f32>,
    pub view_matrix_inverted: Array2<f32>
}

impl WorldPointHandler for ViewMatrixWorldPointHandler {
    fn compute_world_point_from_depth(&self, image_coordinate: &ndarray::Array1<f32>, depth: f32) -> ndarray::Array1<f32> {
        let local_point = self.camera_intrinsics_inverted.dot(&array![image_coordinate[0], image_coordinate[1], 1f32]) * -depth;
        let local_point_swapped_x = array![-local_point[0], local_point[1], local_point[2]];
        let mut world_point_4 = self.view_matrix_inverted.dot(&array![local_point_swapped_x[0], local_point_swapped_x[1], local_point_swapped_x[2], 1f32]);
        world_point_4 /= world_point_4[3];
        
        let mut world_point = Array1::<f32>::zeros(3);
        world_point.assign(&world_point_4.slice(s![..3]));

        world_point
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::world_point_handlers::{view_matrix_world_point_handler::ViewMatrixWorldPointHandler, WorldPointHandler};

    #[test]
    fn compute_world_point_from_depth() {
        let image_point = array![889.63158192f32, 336.58548892f32];
        let depth = 0.5355310460918119f32;
        let camera_intrinsics_inverted = array![[0.00070161547, 0.0, 0.0], [0.0, 0.00070161547, 0.0], [-0.67513853, -0.5045314, 1.0]].t().mapv(|v| v as f32);
        let view_matrix_inverted = array![[-0.9892545, -0.14590749, 0.009293787, 0.0], [0.14538239, -0.9749843, 0.16813457, 0.0], [-0.015470788, 0.16767907, 0.98572004, -0.0], [0.0, 0.0, 0.0, 1.0]].t().mapv(|v| v as f32);

        let world_point_handler = ViewMatrixWorldPointHandler {
            camera_intrinsics_inverted,
            view_matrix_inverted
        };

        assert_eq!(world_point_handler.compute_world_point_from_depth(&image_point, depth), array![0.05617712, -0.22594477, -0.50397223]);
    }
}