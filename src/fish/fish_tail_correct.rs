use geo::{EuclideanDistance, CoordsIter, Point, Polygon};
use geo::algorithm::{ConvexHull, Distance};
use imageproc::contours::find_contours_with_threshold;
use imageproc::point::Point as ImgPoint;

use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use ndarray::prelude::*;

use std::{cmp::Ordering, fmt::Display};
use faer::Mat;
use num::Complex;
use image::{imageops::FilterType, GrayImage, ImageBuffer, Luma, DynamicImage};
use nalgebra::{DMatrix, Matrix2, Vector2};
use ndarray_stats::{errors::EmptyInput, CorrelationExt};

const TARGET_PIXELS: f64 = 30000.0;

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

// need to fix the find countours
// transform fish image to be centered and rotated perfectly
// change x threshold for tail correct (possibly to get x and y also for transformed image)
// scale differently with x and y coordinate to make the convex more promiment in a certian direction

pub struct FishHeadTailDetector;

impl FishHeadTailDetector {
    pub fn find_head_tail(img: &mut ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<(Array1<usize>, Array1<usize>), HeadTailError> {
        let mask: Array2<u8> = Array2::from_shape_vec(
            (img.height() as usize, img.width() as usize),
            img.as_raw().clone(),
        )
        .unwrap();

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

        // println!("{}, {}", coord1, coord2);

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

        let cropped_img = image::imageops::crop_imm(img, x_min as u32, y_min as u32, (x_max - x_min) as u32, (y_max - y_min) as u32).to_image();

        let (width, height) = cropped_img.dimensions();
        let total_pixels = (width as f64) * (height as f64);

        let scale = if total_pixels > TARGET_PIXELS {
            (TARGET_PIXELS / total_pixels).sqrt()
        } else {
            1.0
        };
        println!("{}", width*height);
        println!("{}", scale);

        let new_width = (width as f64 * scale) as u32;
        let new_height = (height as f64 * scale) as u32;

        println!("{}", new_width*new_height);

        let cropped_img = DynamicImage::ImageLuma8(cropped_img)
            .resize_exact(new_width, new_height, FilterType::Lanczos3)
            .to_luma8();

        let scaled_left = array![(left_coord[0] - x_min) as f64 * scale, (left_coord[1] - y_min) as f64 * scale];
        let scaled_right = array![(right_coord[0] - x_min) as f64 * scale, (right_coord[1] - y_min) as f64 * scale];
        let mut tail_coord = array![0.0 as f64, 0.0 as f64];
        let mut head_coord = array![0.0 as f64, 0.0 as f64];
        // get polygon
        if let Some(poly) = extract_polygon(&cropped_img) {
            let hull = poly.convex_hull();
            // distinguish head and tail
            (tail_coord, head_coord) = tail_head_distinct(&hull, &scaled_left, &scaled_right);

            // correct the tail coord
            if let Some(concave_point) = tail_correct(&poly, &hull, &tail_coord) {
                let concave_point_coords = array![
                    ((concave_point.x() /scale) + x_min as f64),
                    ((concave_point.y() /scale) + y_min as f64)
                ];
                draw_dot(img, left_coord[0] as i32, left_coord[1] as i32, 10, Luma([125u8]));
                draw_dot(img, right_coord[0] as i32, right_coord[1] as i32, 10, Luma([100u8]));
                draw_dot(img, concave_point_coords[0] as i32, concave_point_coords[1] as i32, 10, Luma([200u8]));

                tail_coord = concave_point_coords;
            };
        };

        Ok((
            array![(tail_coord[0]).round() as usize, (tail_coord[1]).round() as usize],
            array![(head_coord[0]).round() as usize, (head_coord[1]).round() as usize]
        ))
        

    }
}

fn extract_polygon(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<geo::Polygon<f64>> {

    let contours = find_contours_with_threshold::<u8>(&img, 125);
    if contours.is_empty() {
        return None;
    }

    let largest_contour = contours
        .into_iter()
        .max_by_key(|c| c.points.len())?;
    let exterior: Vec<(f64, f64)> = largest_contour
        .points
        .iter()
        .map(|point: &ImgPoint<u8>| (point.x as f64, point.y as f64))
        .collect();

    Some(Polygon::new(exterior.into(), vec![]))
}

fn tail_head_distinct(hull: &geo::Polygon<f64>, scaled_left: &Array1<f64>, scaled_right: &Array1<f64>)-> (Array1<f64>, Array1<f64>){

    let left_point = Point::new(scaled_left[0], scaled_left[1]);
    let right_point = Point::new(scaled_right[0], scaled_right[1]);

    let left_convexity = hull.exterior().euclidean_distance(&left_point);
    let right_convexity = hull.exterior().euclidean_distance(&right_point);

    if right_convexity < left_convexity {
        (scaled_left.clone(), scaled_right.clone())
    } else {
        (scaled_right.clone(), scaled_left.clone())
    }
}

fn tail_correct(poly: &geo::Polygon<f64>, hull: &geo::Polygon<f64>, left_coord: &Array1<f64>) -> Option<Point<f64>> {
    // println!("HELLOOOO{:?}", hull);
    let mut most_concave_point = None;
    let mut max_concavity = 0.0;

    let left_x = left_coord[0];
    let left_y = left_coord[1];

    let search_radius = 15.0; // THRESHOLD
    let min_x = left_x - search_radius;
    let max_x = left_x + search_radius;

    for point in poly.exterior().coords_iter() {
        let p = Point::new(point.x, point.y);
        let distance_to_hull = hull.exterior().euclidean_distance(&p);

        if point.x >= min_x && point.x <= max_x {
            if distance_to_hull > max_concavity {
                max_concavity = distance_to_hull;
                most_concave_point = Some(p);
            }
        }
    }
    most_concave_point
}



// fn find_most_concave_point(poly: &geo::Polygon<f64>) -> Option<Point<f64>> {
//     let hull = poly.convex_hull();

//     let mut most_concave_point = None;
//     let mut max_distance = 0.0;

//     for point in poly.exterior().coords_iter() {
//         let p = Point::new(point.x, point.y);
//         let distance = hull.exterior().euclidean_distance(&p);

//         if distance > max_distance {
//             max_distance = distance;
//             most_concave_point = Some(p);
//         }
//     }

//     most_concave_point
// }


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
    fn test_fish1() {
        let mut rust_img = image::ImageReader::open("./data/fish1.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish1_out.png").unwrap();
        // assert_eq!(head, array![140, 487]);
        // assert_eq!(tail, array![1873, 406]);
    }
    #[test]
    fn test_fish2() {
        let mut rust_img = image::ImageReader::open("./data/fish2.png").unwrap().decode().unwrap().to_luma8();

        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish2_out.png").unwrap();
        // assert_eq!(head, array![140, 487]);
        // assert_eq!(tail, array![1873, 406]);
    }

    #[test]
    fn test_fish3() {
        let mut rust_img = image::ImageReader::open("./data/fish3.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish3_out.png").unwrap();
    }
    #[test]
    fn test_fish4() {
        let mut rust_img = image::ImageReader::open("./data/fish4.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish4_out.png").unwrap();
    }
    #[test]
    fn test_fish5() {
        let mut rust_img = image::ImageReader::open("./data/fish5.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish5_out.png").unwrap();
    }
    #[test]
    fn test_fish6() {
        let mut rust_img = image::ImageReader::open("./data/fish6.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish6_out.png").unwrap();
    }
    #[test]

    fn test_fish7() {
        let mut rust_img = image::ImageReader::open("./data/fish7.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish7_out.png").unwrap();
    }
    #[test]

    fn test_fish8() {
        let mut rust_img = image::ImageReader::open("./data/fish8.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish8_out.png").unwrap();
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