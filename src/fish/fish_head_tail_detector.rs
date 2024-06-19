use std::{cmp::Ordering, fmt::Display};

use faer::Mat;
use ndarray::{array, s, stack, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
// use ndarray_linalg::Eig;
use ndarray_stats::{errors::EmptyInput, CorrelationExt};
use num::Complex;

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
    pub fn find_head_tail(mask: Array2<u8>) -> Result<(Array1<usize>, Array1<usize>), HeadTailError> {
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
        let x_min = min(&x)? as usize;
        let y_min = min(&y)? as usize;
        let x_max = max(&x)? as usize;
        let y_max = max(&y)? as usize;

        let mask_crop: ArrayBase<ndarray::ViewRepr<&u8>, Dim<[usize; 2]>> = mask.slice(s![y_min..y_max,x_min..x_max]);


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

        let x_mean = (x.iter().sum::<i32>() as f64)/(x.len() as f64);
        let y_mean = (y.iter().sum::<i32>() as f64)/(y.len() as f64);

        let mut new_x = Array1::default(x.len());
        let mut new_y = Array1::default(y.len());

        for i in 0..x.len() {
            new_x[i] = x[i] as f64 - x_mean;
            new_y[i] = y[i] as f64 - y_mean;
        }
        let x_min = new_x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_min = new_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
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

        let eigendecomp: faer::solvers::Eigendecomposition<Complex<f64>> = mat.eigendecomposition();

        let evecs: faer::prelude::MatRef<Complex<f64>> = eigendecomp.u();
        let evals: Vec<Complex<f64>> = mat.eigenvalues();
        let mut real_evals: Vec<f64> = Vec::new();
        
        for i in evals {
            real_evals.push(i.re);
        }
        
        let mut sort_indices = argsort_by(&real_evals, cmp_f64);
        sort_indices.reverse();

        let splice = evecs.get(0..evecs.nrows(), sort_indices[0]).as_2d();

        let (x_v1, y_v1) = (splice.read(0, 0), splice.read(1,0));

        let mask_shape = mask_crop.shape();

        let (height, width) = (mask_shape[0], mask_shape[1]);

        let scale = if height > width {height} else {width};

        let mut coord1: Array1<f64> = array![-x_v1.re * (scale as f64) * 2.0, -y_v1.re * (scale as f64) * 2.0];

        let mut coord2: Array1<f64> = array![x_v1.re * (scale as f64) * 2.0, y_v1.re * (scale as f64) * 2.0];       

        coord1[0] -= x_min;
        coord2[0] += x_min;
        
        coord1[1] -= y_min;
        coord2[1] += y_min;



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

        let coords: ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>> = stack![Axis(0), new_x, new_y];

        let arg_min = match new_x.iter().zip(0..).min() {
            Some((_, min)) => Ok(min),
            None => Err(HeadTailError::MinError)
        }?;

        let arg_max = match new_x.iter().zip(0..).max() {
            Some((_, min)) => Ok(min),
            None => Err(HeadTailError::MaxError)
        }?;

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
        let rust_img = image::io::Reader::open("./data/segmentations.png").unwrap().decode().unwrap().as_luma8().unwrap().clone();
        let mask: ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::from_shape_vec((rust_img.height() as usize, rust_img.width() as usize), rust_img.as_raw().clone()).unwrap();
        let res = FishHeadTailDetector::find_head_tail(mask).unwrap();
        assert_eq!(res, (array![1073, 1114], array![2317,1054]));
    }
}
