use std::{cmp::Ordering, fmt::Display};
use faer::Mat;
use num::Complex;
use image::{GrayImage, Luma};
use ndarray::{Array1, Array2, Axis, s, array};
use nalgebra::{DMatrix, Matrix2, Vector2};
use ndarray_stats::{errors::EmptyInput, CorrelationExt};

#[derive(Debug)]
pub enum HeadTailError {
    MinError,
    MaxError,
    COVError(EmptyInput),
    OOBError,
}

impl Display for HeadTailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeadTailError::MinError => write!(f, "Minimum value of vector could not be computed"),
            HeadTailError::MaxError => write!(f, "Maximum value of vector could not be computed"),
            HeadTailError::COVError(error) => write!(f, "{}", error),
            HeadTailError::OOBError => write!(
                f,
                "Index is out of bounds after argmin/argmax calculation to find coordinate"
            ),
        }
    }
}

pub struct FishHeadTailDetector;

impl FishHeadTailDetector {
    pub fn find_head_tail(mask: &Array2<u8>) -> Result<(Array1<usize>, Array1<usize>), HeadTailError> {
        // Extract non-zero pixel indices
        let nonzero: Vec<(usize, usize)> = mask
            .indexed_iter()
            .filter_map(|((y, x), &val)| if val != 0 { Some((y, x)) } else { None })
            .collect();

        if nonzero.is_empty() {
            return Err(HeadTailError::MinError);
        }

        let y_coords: Array1<usize> = nonzero.iter().map(|&(y, _)| y).collect();
        let x_coords: Array1<usize> = nonzero.iter().map(|&(_, x)| x).collect();

        // Compute bounding box for cropping
        let y_min = min(&y_coords)?;
        let y_max = max(&y_coords)?;
        let x_min = min(&x_coords)?;
        let x_max = max(&x_coords)?;

        let mask_crop = mask.slice(s![y_min..=y_max, x_min..=x_max]);

        // Recalculate non-zero indices within the cropped mask
        let cropped_nonzero: Vec<(usize, usize)> = mask_crop
            .indexed_iter()
            .filter_map(|((y, x), &val)| if val != 0 { Some((y, x)) } else { None })
            .collect();

        if cropped_nonzero.is_empty() {
            return Err(HeadTailError::MinError);
        }

        // Center coordinates
        let cropped_y: Vec<f64> = cropped_nonzero.iter().map(|&(y, _)| y as f64).collect();
        let cropped_x: Vec<f64> = cropped_nonzero.iter().map(|&(_, x)| x as f64).collect();

        let x_mean = mean(&cropped_x);
        let y_mean = mean(&cropped_y);

        let centered_coords: Vec<Vector2<f64>> = cropped_x
            .iter()
            .zip(&cropped_y)
            .map(|(&x, &y)| Vector2::new(x - x_mean, y - y_mean))
            .collect();

        // Calculate covariance matrix
        let covariance_matrix = compute_covariance(&centered_coords)?;

        let eig = covariance_matrix.symmetric_eigen();
        let principal_eigenvector = eig.eigenvectors.column(0);

        let scale = ((mask_crop.nrows().max(mask_crop.ncols())) as f64) * 2.0;
        let scaled_vector = principal_eigenvector * scale;

        let coord1 = Vector2::new(
            -scaled_vector[0] + x_mean,
            -scaled_vector[1] + y_mean,
        );
        let coord2 = Vector2::new(
            scaled_vector[0] + x_mean,
            scaled_vector[1] + y_mean,
        );

        println!("{}, {}", coord1, coord2);

        let m = principal_eigenvector[1] / principal_eigenvector[0];
        let b = coord1[1] - m * coord1[0];

        let mut y_target: Array1<f64> = Array1::default(cropped_x.len());
        for i in 0..cropped_x.len() {
            y_target[i] = m * cropped_x[i] + b;
        }

        let mut y_abs_diff: Vec<usize> = Vec::new();
        for (i, &val) in cropped_y.iter().enumerate() {
            if (val - y_target[i]).abs() < 1.0 {
                y_abs_diff.push(i);
            }
        }

        let mut new_x: Array1<usize> = Array1::default(y_abs_diff.len());
        let mut new_y: Array1<usize> = Array1::default(y_abs_diff.len());
        for (i, &idx) in y_abs_diff.iter().enumerate() {
            new_x[i] = cropped_nonzero[idx].1;
            new_y[i] = cropped_nonzero[idx].0;
        }

        let arg_min = new_x
            .iter()
            .enumerate()
            .min_by_key(|&(_, &val)| val)
            .map(|(idx, _)| idx)
            .ok_or(HeadTailError::MinError)?;

        let arg_max = new_x
            .iter()
            .enumerate()
            .max_by_key(|&(_, &val)| val)
            .map(|(idx, _)| idx)
            .ok_or(HeadTailError::MaxError)?;

        let left_coord = array![new_x[arg_min] + x_min, new_y[arg_min] + y_min];
        let right_coord = array![new_x[arg_max] + x_min, new_y[arg_max] + y_min];

        Ok((left_coord, right_coord))

    }
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn min<T: Ord + Copy>(array: &Array1<T>) -> Result<T, HeadTailError> {
    array
        .iter()
        .copied()
        .min()
        .ok_or(HeadTailError::MinError)
}

fn max<T: Ord + Copy>(array: &Array1<T>) -> Result<T, HeadTailError> {
    array
        .iter()
        .copied()
        .max()
        .ok_or(HeadTailError::MaxError)
}

fn compute_covariance(coords: &[Vector2<f64>]) -> Result<Matrix2<f64>, HeadTailError> {
    let n = coords.len() as f64;

    let sum_xx = coords.iter().map(|v| v[0] * v[0]).sum::<f64>();
    let sum_xy = coords.iter().map(|v| v[0] * v[1]).sum::<f64>();
    let sum_yy = coords.iter().map(|v| v[1] * v[1]).sum::<f64>();

    let covariance_matrix = Matrix2::new(sum_xx / n, sum_xy / n, sum_xy / n, sum_yy / n);

    Ok(covariance_matrix)
}

// Test function
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_head_tail() {
        let mut rust_img = image::ImageReader::open("./data/fish.png").unwrap().decode().unwrap().to_luma8();
        let mask: Array2<u8> = Array2::from_shape_vec(
            (rust_img.height() as usize, rust_img.width() as usize),
            rust_img.as_raw().clone(),
        )
        .unwrap();

        let (head, tail) = FishHeadTailDetector::find_head_tail(&mask).unwrap();
        println!("{}", head);
        println!("{}", tail);

        draw_dot(&mut rust_img, head[0] as i32, head[1] as i32, 10, Luma([125u8]));
        draw_dot(&mut rust_img, tail[0] as i32, tail[1] as i32, 10, Luma([200u8]));

        rust_img.save("./data/fish_processed.png").unwrap();
        assert_eq!(head, array![140, 487]);
        assert_eq!(tail, array![1873, 406]);
    }
    #[test]
    fn test_tail_correction() {
        let mut rust_img = image::ImageReader::open("./data/fish_tail.png").unwrap().decode().unwrap().to_luma8();
        let mask: Array2<u8> = Array2::from_shape_vec(
            (rust_img.height() as usize, rust_img.width() as usize),
            rust_img.as_raw().clone(),
        )
        .unwrap();

        let (head, tail) = FishHeadTailDetector::find_head_tail(&mask).unwrap();
        println!("{}", head);
        println!("{}", tail);

        draw_dot(&mut rust_img, head[0] as i32, head[1] as i32, 10, Luma([125u8]));
        draw_dot(&mut rust_img, tail[0] as i32, tail[1] as i32, 10, Luma([200u8]));

        rust_img.save("./data/fish_tail_processed.png").unwrap();
        assert_eq!(head, array![140, 487]);
        assert_eq!(tail, array![1873, 406]);
    }
}

fn draw_dot(image: &mut GrayImage, x: i32, y: i32, radius: i32, color: Luma<u8>) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x_pos = x + dx;
                let y_pos = y + dy;
                if x_pos >= 0 && y_pos >= 0 && x_pos < image.width() as i32 && y_pos < image.height() as i32 {
                    image.put_pixel(x_pos as u32, y_pos as u32, color);
                }
            }
        }
    }
}