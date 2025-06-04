use ndarray::Array1;

pub fn norm(array: &Array1<f32>) -> f32 {
    (array[0] * array[0] + array[1] * array[1]).sqrt()
}

