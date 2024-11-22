use std::{cmp::Ordering, fmt::Display};

use faer::Mat;
use ndarray::{array, s, stack, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
// use ndarray_linalg::Eig;
use ndarray_stats::{errors::EmptyInput, CorrelationExt};
use num::Complex;
use image::{Luma, GrayImage};


#[derive(Debug)]
pub enum HeadTailError {
    MinError,
    MaxError,
    COVError(EmptyInput),
    OOBError
}

impl Display for HeadTailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeadTailError::MinError => 
                write!(f, "Minimum value of vector could not be computed"),
            HeadTailError::MaxError => 
                write!(f, "Maximum value of vector could not be computed"),
            HeadTailError::COVError(error) => 
                write!(f, "{}", error),
            HeadTailError::OOBError =>
                write!(f, "Index is out of bounds after argmin/argmax calculation to find coordinate")
        }
    }
}

pub struct FishHeadTailDetector {

}

impl FishHeadTailDetector {
    pub fn find_head_tail(mask: &Array2<u8>) -> Result<(Array1<usize>, Array1<usize>), HeadTailError> {
        // Nonzero pixels from mask
        let nonzero: Array1<(i32, i32)> = mask
                .indexed_iter()
                .filter_map(|(index, &item)| if item != 0 {Some((index.0 as i32, index.1 as i32))} else { None })
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

        let mask_crop: ArrayBase<ndarray::ViewRepr<&u8>, Dim<[usize; 2]>> = mask.slice(s![y_min..y_max,x_min..x_max]);
        
        // Recalculate non-zero indices after cropping
        let nonzero_inds: Array1<(i32, i32)> = mask_crop
                .indexed_iter()
                .filter_map(|(index, &item)| if item != 0 {Some((index.0 as i32, index.1 as i32))} else { None })
                .collect();        
        
        let mut y = Array1::default(nonzero_inds.len());
        let mut x = Array1::default(nonzero_inds.len());

        for i in 0..nonzero_inds.len() {
            y[i] = nonzero_inds[i].0;
            x[i] = nonzero_inds[i].1;
        }

        // Centering the Data
        // let x_mean = (x.iter().sum::<i32>() as f64)/(x.len() as f64);
        // let y_mean = (y.iter().sum::<i32>() as f64)/(y.len() as f64);
        let x_mean = (x.iter().map(|&val| val as i64).sum::<i64>() as f64) / (x.len() as f64);
        let y_mean = (y.iter().map(|&val| val as i64).sum::<i64>() as f64) / (y.len() as f64);
        println!("{} {}", x_mean, y_mean);

        let mut new_x = Array1::default(x.len());
        let mut new_y = Array1::default(y.len());

        for i in 0..x.len() {
            new_x[i] = x[i] as f64 - x_mean;
            new_y[i] = y[i] as f64 - y_mean;
        }
        let x_min = new_x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_min = new_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        println!("{} {}", x_min, y_min);

        // Covariance Matrix Calculation
        let coords: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = stack![Axis(0), new_x, new_y];

        let cov = match CorrelationExt::cov(&coords, 1.0) {
            Ok(matrix) => Ok(matrix),
            Err(error) => Err(HeadTailError::COVError(error))
        }?;

        let mut mat: Mat<f64> = Mat::zeros(2, cov.select(Axis(0), &[0]).len());

        for i in cov.indexed_iter() {
            match i {
                (loc, val) => mat[loc] = *val,
            }
        }
        
        // Eigenvalue Decomposition

        let eigendecomp: faer::solvers::Eigendecomposition<Complex<f64>> = mat.eigendecomposition();

        let evecs: faer::prelude::MatRef<Complex<f64>> = eigendecomp.u();
        let evals: Vec<Complex<f64>> = mat.eigenvalues();
        let mut real_evals: Vec<f64> = Vec::new();
        
        for i in evals {
            real_evals.push(i.re);
        }
        
        // Calculate Head and Tail Coordinates

        let mut sort_indices = argsort_by(&real_evals, cmp_f64);
        sort_indices.reverse();

        let splice = evecs.get(0..evecs.nrows(), sort_indices[0]).as_2d();

        let (x_v1, y_v1) = (splice.read(0, 0), splice.read(1,0));

        let mask_shape = mask_crop.shape();

        let (height, width) = (mask_shape[0], mask_shape[1]);

        let scale = if height > width {height} else {width};

        let mut coord1: Array1<f64> = array![-x_v1.re * (scale as f64) * 2.0, -y_v1.re * (scale as f64) * 2.0];

        let mut coord2: Array1<f64> = array![x_v1.re * (scale as f64) * 2.0, y_v1.re * (scale as f64) * 2.0];       

        // Adjust coordinates back to original space
        coord1[0] -= x_min;
        coord2[0] += x_min;
        
        coord1[1] -= y_min;
        coord2[1] += y_min;

        println!("{} {}", coord1[1], coord1[0]);

        // Line equation calculation for coord1
        let m = y_v1.re / x_v1.re;
        let b = coord1[1] - m * coord1[0];

        let mut y_target: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array1::default(x.len());

        for i in 0..x.len() {
            y_target[i] = m * (x[i] as f64) + b;
        }
        let mut y_abs_diff : Vec<usize> = Vec::default();
        for i in 0..y.len() {
            if (y[i] as f64 - y_target[i]).abs() < 1.0 {
                y_abs_diff.push(i)
            }
        }

        let mut new_x: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = Array1::default(y_abs_diff.len());
        let mut new_y: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = Array1::default(y_abs_diff.len());
        
        for i in 0..y_abs_diff.len() {
            new_x[i] = x[y_abs_diff[i]] as usize;
            new_y[i] = y[y_abs_diff[i]] as usize;
        }

        // Find minimum and maximum x coord extremes
        let coords: ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>> = stack![Axis(0), new_x, new_y];

        let arg_min = match new_x.iter().zip(0..).min() {
            Some((_, min)) => Ok(min),
            None => Err(HeadTailError::MinError)
        }?;

        let arg_max = match new_x.iter().zip(0..).max() {
            Some((_, min)) => Ok(min),
            None => Err(HeadTailError::MaxError)
        }?;

        // extract head and tail coordinates
        let mut left_coord: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = array![get(&coords, (0, arg_min))?, get(&coords, (1, arg_min))?];
        let mut right_coord = array![get(&coords, (0, arg_max))?, get(&coords, (1, arg_max))?];

        let mut y = Array1::default(nonzero.len());
        let mut x = Array1::default(nonzero.len());
        
        
        for i in 0..nonzero.len() {
            y[i] = nonzero[i].0;
            x[i] = nonzero[i].1;
        }

        let x_min = min(&x)? as usize;
        let y_min = min(&y)? as usize;

        // Adjust coordinates back to original space
        left_coord[0] += x_min;
        right_coord[0] += x_min;
        left_coord[1] += y_min;
        right_coord[1] += y_min;
        
        Ok((left_coord, right_coord))
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
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

fn min <T:Ord + Copy> (array: &Array1<T>) -> Result<T, HeadTailError> {
    match (*array).iter().min() {
        None => Err(HeadTailError::MinError),
        Some(t) => Ok(*t)
    }    
}

fn max <T:Ord + Copy> (array: &Array1<T>) -> Result<T, HeadTailError> {
    match (*array).iter().max() {
        None => Err(HeadTailError::MaxError),
        Some(t) => Ok(*t)
    }    
}

fn get(array: &ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>>, index: (usize, usize)) -> Result<usize, HeadTailError> {
    match array.get(index) {
        None => Err(HeadTailError::OOBError),
        Some(coord) => Ok(*coord)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // let rust_img = image::ImageReader::open("./data/fish.png").unwrap().decode().unwrap().as_luma8().unwrap().clone();

        let mut rust_img = image::ImageReader::open("./data/fish.png").unwrap().decode().unwrap().to_luma8().clone();
        // rust_img.save("./data/processed_image.png");
        let mask: ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::from_shape_vec((rust_img.height() as usize, rust_img.width() as usize), rust_img.as_raw().clone()).unwrap();
        let res = FishHeadTailDetector::find_head_tail(&mask).unwrap();
        // let res = (array![20, 100], array![202,100]);
        let head_coord = res.0;
        let tail_coord = res.1;
    
        draw_dot(&mut rust_img, head_coord[0] as i32, head_coord[1] as i32, 10, Luma([125u8]));
        draw_dot(&mut rust_img, tail_coord[0] as i32, tail_coord[1] as i32, 10, Luma([200u8]));
    
        rust_img.save("./data/processed_image.png");

        // assert_eq!(res, (array![1073, 1114], array![2317,1054]));


    }
}
fn draw_dot(image: &mut GrayImage, x_center: i32, y_center: i32, radius: i32, color: Luma<u8>) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                let x_pos = x_center + x;
                let y_pos = y_center + y;
                if x_pos >= 0 && y_pos >= 0 && x_pos < image.width() as i32 && y_pos < image.height() as i32 {
                    image.put_pixel(x_pos as u32, y_pos as u32, color);
                }
            }
        }
    }
}