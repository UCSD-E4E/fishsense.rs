use geo::{EuclideanDistance,ConvexHull, Line, CoordsIter, Point, Polygon};

// use geo::algorithm::{ConvexHull, Distance, BoundingRect};
// use imageproc::contours::find_contours_with_threshold;
// use imageproc::point::Point as ImgPoint;
// use ndarray::linalg::Dot;

use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use ndarray::prelude::*;

use std::{cmp::Ordering, error::Error, fmt::Display};
// use faer::Mat;
// use num::Complex;
use image::{imageops::FilterType, GrayImage, ImageBuffer, Luma, DynamicImage};
use nalgebra::{DMatrix, Matrix2, Vector2};
use ndarray_stats::{errors::EmptyInput, CorrelationExt};
use anyhow::{Result,};

use opencv::{
    core::{Mat,Scalar, Point as CVPoint},
    imgproc,
    prelude::*,
    types,
};


// const TARGET_PIXELS: f64 = 30000.0;
const TARGET_PIXELS: f64 = 35000.0;

#[derive(Debug)]
pub enum HeadTailError {
    MinError,
    MaxError,
    COVError(EmptyInput),
    OOBError,
    PolygonError,
    
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
            HeadTailError::PolygonError => write!(f, "Polygon did not extract, lower target pixels"),
        }
    }
}
impl Error for HeadTailError {
    // Optionally, you can override `source` to return inner errors
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            HeadTailError::COVError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<anyhow::Error> for HeadTailError {
    fn from(_: anyhow::Error) -> Self {
        HeadTailError::PolygonError
    }
}


// draw contours with imageproc (removed dependancy)
// smoothing?
// better error handling
// extend custom libraries and remove redundant computations
// remove outdated dependancies

// optional: use original python head tail distinct process with polygon differences
// optional: work with concave tails and ensure head vs tail distinguish works

pub struct FishHeadTailDetector;

impl FishHeadTailDetector {
    pub fn find_head_tail(img: &mut ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<(Array1<usize>, Array1<usize>), HeadTailError> {
        let mask: Array2<u8> = Array2::from_shape_vec(
            (img.height() as usize, img.width() as usize),
            img.as_raw().clone(),
        )
        .unwrap();

        // non-zero pixel idxs
        let nonzero: Vec<(usize, usize)> = mask
            .indexed_iter()
            .filter_map(|((y, x), &val)| if val != 0 { Some((y, x)) } else { None })
            .collect();

        if nonzero.is_empty() {
            return Err(HeadTailError::MinError);
        }

        let y_coords: Array1<usize> = nonzero.iter().map(|&(y, _)| y).collect();
        let x_coords: Array1<usize> = nonzero.iter().map(|&(_, x)| x).collect();

        // Calc bounding box for cropping
        let y_min = min(&y_coords)?;
        let y_max = max(&y_coords)?;
        let x_min = min(&x_coords)?;
        let x_max = max(&x_coords)?;

        let mask_crop = mask.slice(s![y_min..=y_max, x_min..=x_max]);

        // Non-zero idxs in cropped mask
        let cropped_nonzero: Vec<(usize, usize)> = mask_crop
            .indexed_iter()
            .filter_map(|((y, x), &val)| if val != 0 { Some((y, x)) } else { None })
            .collect();

        if cropped_nonzero.is_empty() {
            return Err(HeadTailError::MinError);
        }

        // center coords
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
        // let coord2 = Vector2::new(
        //     scaled_vector[0] + x_mean,
        //     scaled_vector[1] + y_mean,
        // );

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

        draw_dot(img, left_coord[0] as i32, left_coord[1] as i32, 10, Luma([190u8]));
        draw_dot(img, right_coord[0] as i32, right_coord[1] as i32, 10, Luma([190u8]));

        let (width, height) = cropped_img.dimensions();
        let total_pixels = (width as f64) * (height as f64);

        let scale = if total_pixels > TARGET_PIXELS {
            (TARGET_PIXELS / total_pixels).sqrt()
        } else {
            1.0
        };

        let new_width = (width as f64 * scale) as u32;
        let new_height = (height as f64 * scale) as u32;

        let cropped_img = DynamicImage::ImageLuma8(cropped_img)
            .resize_exact(new_width, new_height, FilterType::Lanczos3)
            .to_luma8();

        let scaled_left = array![(left_coord[0] - x_min) as f64 * scale, (left_coord[1] - y_min) as f64 * scale];
        let scaled_right = array![(right_coord[0] - x_min) as f64 * scale, (right_coord[1] - y_min) as f64 * scale];
        let mut tail_coord = array![0.0 as f64, 0.0 as f64];
        let mut head_coord = array![0.0 as f64, 0.0 as f64];
        // get polygon
        match extract_polygon(&cropped_img)? {
            Some(poly) => {

                let hull = poly.convex_hull();

                // distinguish head and tail
                (tail_coord, head_coord) = tail_head_distinct(&hull, &scaled_left, &scaled_right);

                // ab for head correct
                let ab = Vector2::new(
                    head_coord[0] - tail_coord[0],
                    head_coord[1] - tail_coord[1],
                );
                let ab_perp = Vector2::new(-ab.y, ab.x);

                let search_radius = ab.norm()*0.09;

                // correct the tail coord
                if let Some(concave_point) = tail_correct(&poly, &hull, &tail_coord, search_radius) {

                    tail_coord = array![
                        ((concave_point.x() /scale) + x_min as f64),
                        ((concave_point.y() /scale) + y_min as f64)
                    ];

                    draw_dot(img, tail_coord[0] as i32, tail_coord[1] as i32, 10, Luma([100u8]));
                };
                // correct the head coord

                if let Some(correct_head) = head_correct(&hull, &head_coord, &ab, &ab_perp) {
                    head_coord = array![
                        (correct_head.x() / scale) + x_min as f64,
                        (correct_head.y() / scale) + y_min as f64
                    ];
                    draw_dot(img, head_coord[0] as i32, head_coord[1] as i32, 3, Luma([50u8]));

                };
            }
            None => {
                return Err(HeadTailError::PolygonError);
            }

        }

        Ok((
            array![(tail_coord[0]).round() as usize, (tail_coord[1]).round() as usize],
            array![(head_coord[0]).round() as usize, (head_coord[1]).round() as usize]
        ))
        

    }
}


pub fn extract_polygon(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> anyhow::Result<Option<Polygon<f64>>> {
    let (width, height) = (img.width() as i32, img.height() as i32);

    let raw_slice = img.as_raw();

    let mat_from_slice = Mat::from_slice(raw_slice)?;

    let mat = mat_from_slice.reshape(1, height)?;

    let mut thresh = Mat::default();
    imgproc::threshold(&mat, &mut thresh, 125.0, 255.0, imgproc::THRESH_BINARY)?;

    let mut contours = types::VectorOfVectorOfPoint::new();
    imgproc::find_contours(
        &thresh,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        CVPoint::new(0, 0),
    )?;

    if contours.len() == 0 {
        return Ok(None);
    }

    let mut max_area = 0.0;
    let mut max_contour = None;
    for contour in contours.iter() {
        let area = imgproc::contour_area(&contour, false)?;
        if area > max_area {
            max_area = area;
            max_contour = Some(contour.clone());
        }
    }

    let contour = match max_contour {
        Some(c) => c,
        None => return Ok(None),
    };

    let exterior: Vec<_> = contour
        .iter()
        .map(|pt| (pt.x as f64, pt.y as f64))
        .collect();

    Ok(Some(Polygon::new(exterior.into(), vec![])))
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


fn tail_correct(
    poly: &geo::Polygon<f64>,
    hull: &geo::Polygon<f64>,
    left_coord: &Array1<f64>,
    search_radius: f64
) -> Option<Point<f64>> {
    let mut most_concave_point = None;
    let mut max_concavity = 0.0;

    let left_point = Point::new(left_coord[0], left_coord[1]);
    // let search_radius = 20.0; // Static radius

    let coords: Vec<_> = poly.exterior().coords_iter().collect();
    let n = coords.len();

    for (i, coord) in coords.iter().enumerate() {
        let p = Point::new(coord.x, coord.y);
        let distance_to_left = left_point.euclidean_distance(&p);
        if distance_to_left > search_radius {
            continue;
        }

        let distance_to_hull = hull.exterior().euclidean_distance(&p);

        // get prev and next points
        let prev = Point::new(coords[(i + n - 1) % n].x, coords[(i + n - 1) % n].y);
        let next = Point::new(coords[(i + 1) % n].x, coords[(i + 1) % n].y);

        let prev_distance = hull.exterior().euclidean_distance(&prev);
        let next_distance = hull.exterior().euclidean_distance(&next);

        // local minimum (deeper than neighbors)
        if distance_to_hull > prev_distance && distance_to_hull > next_distance {
            if distance_to_hull > max_concavity {
                max_concavity = distance_to_hull;
                most_concave_point = Some(p);
            }
        }
    }

    most_concave_point
}

fn head_correct(
    hull: &geo::Polygon<f64>,
    head_coord: &Array1<f64>,
    ab: &Vector2<f64>,
    ab_perp: &Vector2<f64>
) -> Option<Point<f64>> {

    // ab_perp to line centered at the head
    let p1 = Point::new(
        head_coord[0] - ab_perp[0],
        head_coord[1] - ab_perp[1],
    );
    let p2 = Point::new(
        head_coord[0] + ab_perp[0],
        head_coord[1] + ab_perp[1],
    );
    let perp_line = Line::new(p1, p2);

    // dir vector from tail to head
    let ab_dir = ab.normalize();

    let mut max_dist = -1.0;
    let mut best_point = Some(Point::new(head_coord[0], head_coord[1]));

    for point in hull.exterior().points_iter() {
        let vec_to_point = Vector2::new(point.x() - head_coord[0], point.y() - head_coord[1]);
        let projection = vec_to_point.dot(&ab_dir);

        // skip points that lie in the neg dir of ab
        if -1.0*projection > 0.0 {
            continue;
        }

        let dist = perp_line.euclidean_distance(&point);
        if dist > max_dist {
            max_dist = dist;
            best_point = Some(point);
        }
    }

    best_point
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

fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    let dot = v1.dot(v2);
    let norm_v1 = v1.dot(v1).sqrt();
    let norm_v2 = v2.dot(v2).sqrt();
    dot / (norm_v1 * norm_v2)
}

// Test function
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fish1() {
        println!("fish1");
        let mut rust_img = image::ImageReader::open("./data/fish1.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish1_out.png").unwrap();
        // assert_eq!(head, array![140, 487]);
        // assert_eq!(tail, array![1873, 406]);
    }
    #[test]
    fn test_fish2() {
        println!("fish2");
        let mut rust_img = image::ImageReader::open("./data/fish2.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish2_out.png").unwrap();
        // assert_eq!(head, array![140, 487]);
        // assert_eq!(tail, array![1873, 406]);
    }

    #[test]
    fn test_fish3() {
        println!("fish3");
        let mut rust_img = image::ImageReader::open("./data/fish3.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish3_out.png").unwrap();
    }
    #[test]
    fn test_fish4() {
        println!("fish4");
        use std::time::Instant;

    let start = Instant::now();

        let mut rust_img = image::ImageReader::open("./data/fish4.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        // rust_img.save("./data/fish4_out.png").unwrap();
                        let duration = start.elapsed(); 
    println!("Test completed in: {:?}", duration);
    }
    #[test]
    fn test_fish5() {
        println!("fish5");
        let mut rust_img = image::ImageReader::open("./data/fish5.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish5_out.png").unwrap();
    }
    #[test]
    fn test_fish6() {
        println!("fish6");
        let mut rust_img = image::ImageReader::open("./data/fish6.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish6_out.png").unwrap();
    }
    #[test]

    fn test_fish7() {
        println!("fish7");
        let mut rust_img = image::ImageReader::open("./data/fish7.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish7_out.png").unwrap();
    }
    #[test]

    fn test_fish8() {
        println!("fish8");
        let mut rust_img = image::ImageReader::open("./data/fish8.png").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish8_out.png").unwrap();
    }
    #[test]

    fn test_fish9() {
        println!("fish9");
        let mut rust_img = image::ImageReader::open("./data/fish9.jpeg").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/fish9_out.png").unwrap();
    }
    #[test]
    fn test_fish10() {
        println!("seg");
        let mut rust_img = image::ImageReader::open("./data/test1_seg.jpeg").unwrap().decode().unwrap().to_luma8();
        let (head, tail) = FishHeadTailDetector::find_head_tail(&mut rust_img).unwrap();
        rust_img.save("./data/test1_out.png").unwrap();
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

fn draw_convex_hull_points(
    img: &mut GrayImage,
    hull: &Polygon<f64>,
    x_min: i32,
    y_min: i32,
    scale: f64,
    radius: i32,
    color: Luma<u8>,
) {
    for coord in hull.exterior().coords_iter() {
        let x = ((coord.x / scale) + x_min as f64).round() as i32;
        let y = ((coord.y / scale) + y_min as f64).round() as i32;
        draw_dot(img, x, y, radius, color);
    }
}

fn draw_perpendicular_line(
    img: &mut GrayImage,
    center: &Array1<f64>,
    direction: &Vector2<f64>,
    length: f64,
    step: f64,
    radius: i32,
    color: Luma<u8>,
    x_min: i32,
    y_min: i32,
    scale: f64,
) {
    let dir_norm = direction.normalize();
    let half_len = length / 2.0;

    let num_steps = (length / step).ceil() as i32;

    for i in -num_steps..=num_steps {
        let offset = i as f64 * step;
        let point = array![
            center[0] + dir_norm.x * offset,
            center[1] + dir_norm.y * offset
        ];

        let x = ((point[0] / scale) + x_min as f64).round() as i32;
        let y = ((point[1] / scale) + y_min as f64).round() as i32;

        draw_dot(img, x, y, radius, color);
    }
}