// use std::ffi::c_void;

// use opencv::core::{Mat, CV_8UC1};
// use ndarray::Array2;

// pub fn as_mat_u8c1_mut(array: &Array2<u8>) -> Result<Mat, opencv::Error> {
//     // Array must be contiguous and in the standard row-major layout, or the
//     // conversion to a `Mat` will produce a corrupted result
//     assert!(array.is_standard_layout());

//     let (height, width) = array.dim();
//     let array_clone = array.clone();
//     unsafe { Mat::new_rows_cols_with_data_unsafe_def(
//         height as i32,
//         width as i32,
//         CV_8UC1,
//         array_clone.into_raw_vec().as_ptr() as *mut c_void,
//     ) }
// }