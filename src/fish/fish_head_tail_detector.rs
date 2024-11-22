use std::{cmp::Ordering, fmt::Display};
use faer::Mat;
use ndarray::{array, s, stack, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_stats::{errors::EmptyInput, CorrelationExt};
use nalgebra::DMatrix;
use num::Complex;
use image::{GrayImage, Luma};

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
        // Nonzero pixels from mask
        let nonzero: Array1<(i32, i32)> = mask
            .indexed_iter()
            .filter_map(|(index, &item)| if item != 0 { Some((index.0 as i32, index.1 as i32)) } else { None })
            .collect();

        let mut y = Array1::default(nonzero.len());
        let mut x = Array1::default(nonzero.len());

        for i in 0..nonzero.len() {
            y[i] = nonzero[i].0;
            x[i] = nonzero[i].1;
        }

        // Compute min/max coordinates for cropping
        let x_min = min(&x)? as usize;
        let y_min = min(&y)? as usize;
        let x_max = max(&x)? as usize;
        let y_max = max(&y)? as usize;

        let mask_crop: ArrayBase<ndarray::ViewRepr<&u8>, Dim<[usize; 2]>> = mask.slice(s![y_min..y_max, x_min..x_max]);

        // Recalculate non-zero indices after cropping
        let nonzero_inds: Array1<(i32, i32)> = mask_crop
            .indexed_iter()
            .filter_map(|(index, &item)| if item != 0 { Some((index.0 as i32, index.1 as i32)) } else { None })
            .collect();

        let mut y = Array1::default(nonzero_inds.len());
        let mut x = Array1::default(nonzero_inds.len());

        for i in 0..nonzero_inds.len() {
            y[i] = nonzero_inds[i].0;
            x[i] = nonzero_inds[i].1;
        }

        // Centering the Data
        let x_mean = (x.iter().map(|&val| val as i64).sum::<i64>() as f64) / (x.len() as f64);
        let y_mean = (y.iter().map(|&val| val as i64).sum::<i64>() as f64) / (y.len() as f64);

        let new_x: Array1<f64> = x.iter().map(|&xi| xi as f64 - x_mean).collect();
        let new_y: Array1<f64> = y.iter().map(|&yi| yi as f64 - y_mean).collect();

        let x_min = new_x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_min = new_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        // Covariance Matrix Calculation
        let coords: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = stack![Axis(0), new_x, new_y];
        let cov = match CorrelationExt::cov(&coords, 1.0) {
            Ok(matrix) => Ok(matrix),
            Err(error) => Err(HeadTailError::COVError(error)),
        }?;

        // Convert ndarray covariance matrix to nalgebra's DMatrix
        let cov_shape = cov.shape();
        let rows = cov_shape[0];
        let cols = cov_shape[1];
        let cov_matrix = DMatrix::from_vec(rows, cols, cov.as_slice().unwrap().to_vec());

        // Eigenvalue Decomposition
        let eigen = nalgebra::linalg::SymmetricEigen::new(cov_matrix);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Select eigenvector corresponding to the largest eigenvalue
        let mut sort_indices = argsort_by(&eigenvalues.iter().map(|x| *x).collect::<Vec<_>>(), cmp_f64);
        sort_indices.reverse();
        let principal_eigenvector = eigenvectors.column(sort_indices[0]);

        let scale = mask_crop.shape()[0].max(mask_crop.shape()[1]) as f64;
        let (x_v, y_v) = (principal_eigenvector[0], principal_eigenvector[1]);
        let head = array![x_mean + x_v * scale, y_mean + y_v * scale];
        let tail = array![x_mean - x_v * scale, y_mean - y_v * scale];

        Ok((head.mapv(|v| v as usize), tail.mapv(|v| v as usize)))

    }
}

fn argsort_by<T, F>(data: &[T], mut compare: F) -> Vec<usize>
where
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| compare(&data[i], &data[j]));
    indices
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        return Ordering::Greater;
    }
    if b.is_nan() {
        return Ordering::Less;
    }
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn min<T: Ord + Copy>(array: &Array1<T>) -> Result<T, HeadTailError> {
    match (*array).iter().min() {
        None => Err(HeadTailError::MinError),
        Some(t) => Ok(*t),
    }
}

fn max<T: Ord + Copy>(array: &Array1<T>) -> Result<T, HeadTailError> {
    match (*array).iter().max() {
        None => Err(HeadTailError::MaxError),
        Some(t) => Ok(*t),
    }
}

fn get(array: &ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>>, index: (usize, usize)) -> Result<usize, HeadTailError> {
    match array.get(index) {
        None => Err(HeadTailError::OOBError),
        Some(coord) => Ok(*coord)
    }
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

        rust_img.save("./data/processed_image.png").unwrap();
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