use ndarray::Array1;

pub trait WorldPointHandler {
    fn compute_world_point_from_depth(&self, image_coordinate: &Array1<f32>, depth: f32) -> Array1<f32>;
}