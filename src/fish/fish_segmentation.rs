use std::cmp::{max, min};
use std::fmt::Display;
use std::fs::{File, create_dir};
use std::io::{Cursor, copy};
use std::path::PathBuf;

use app_dirs2::{AppDataType, AppInfo, app_root};
use bytes::Bytes;
use ndarray::{s, Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use opencv::core::{Mat, Point2i, Size, VectorToVec, CV_8UC1};
use opencv::imgproc::{fill_poly, find_contours_with_hierarchy, resize_def, CHAIN_APPROX_NONE, LINE_8, RETR_CCOMP};
use opencv::types::{VectorOfPoint, VectorOfVec4i, VectorOfVectorOfPoint};
use ort::Session;
use reqwest::blocking::get;
use cv_convert::TryIntoCv;

#[derive(Debug)]
pub enum SegmentationError {
    AppDirsError(app_dirs2::AppDirsError),
    ArrayShapeError(ndarray::ShapeError),
    CopyErr(u64),
    CVToNDArrayError,
    DownloadError(reqwest::Error),
    FishNotFound,
    IOError(std::io::Error),
    ModelLoadError,
    NDArrayToCVError,
    OpenCVError(opencv::Error),
    OrtErr(ort::Error),
    PolyNotFound,
}

impl Display for SegmentationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentationError::AppDirsError(error) => 
                write!(f, "{}", error),
            SegmentationError::ArrayShapeError(error) => 
                write!(f, "{}", error),
            SegmentationError::CopyErr(error) => 
                write!(f, "{}", error),
            SegmentationError::CVToNDArrayError => 
                write!(f, "CVToNDArrayError"),
            SegmentationError::DownloadError(error) => 
                write!(f, "{}", error),
            SegmentationError::FishNotFound => 
                write!(f, "FishNotFound"),
            SegmentationError::IOError(error) => 
                write!(f, "{}", error),
            SegmentationError::ModelLoadError => 
                write!(f, "ModelLoadError"),
            SegmentationError::NDArrayToCVError => 
                write!(f, "NDArrayToCVError"),
            SegmentationError::OpenCVError(error) => 
                write!(f, "{}", error),
            SegmentationError::OrtErr(error) => 
                write!(f, "{}", error),
            SegmentationError::PolyNotFound => 
                write!(f, "PolyNotFound")
        }
    }
}

pub struct FishSegmentation {
    model_path: PathBuf,
    model_set: bool,
    model: Option<Session>,
}

impl FishSegmentation {
    const MIN_SIZE_TEST: usize = 800;
    const MAX_SIZE_TEST: usize = 1058;

    const SCORE_THRESHOLD: f32 = 0.3;
    const MASK_THRESHOLD: f32 = 0.5;

    const MODEL_URL: &'static str = "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true";

    fn get_model_path() -> Result<PathBuf, SegmentationError> {
        match app_root(AppDataType::UserCache, &AppInfo{
            name: "fishsense.rs",
            author: "Engineers for Exploration"
        }) {
            Ok(mut path) => {
                path.push("models");

                if !path.exists() {
                    match create_dir(&path) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(SegmentationError::IOError(err))
                    }?;
                }

                path.push("fishial.onnx");
        
                Ok(path)
            }
            Err(err) => Err(SegmentationError::AppDirsError(err))
        }
    }

    fn download_model_bytes() -> Result<Bytes, SegmentationError> {
        match get(FishSegmentation::MODEL_URL) {
            Ok(response) => {
                match response.bytes() {
                    Ok(bytes) => Ok(bytes),
                    Err(err) => Err(SegmentationError::DownloadError(err))
                }
            }
            Err(err) => Err(SegmentationError::DownloadError(err))
        }
    }

    fn save_bytes_to_file(bytes: &Bytes, path: &PathBuf) -> Result<u64, SegmentationError> {
        match File::create(path) {
            Ok(mut file) => {
                let mut content =  Cursor::new(bytes);

                match copy(&mut content, &mut file) {
                    Ok(byte_count) => Ok(byte_count),
                    Err(err) => Err(SegmentationError::IOError(err))
                }
            }
            Err(err) => Err(SegmentationError::IOError(err))
        }
    }

    fn download_model() -> Result<PathBuf, SegmentationError> {
        let path = FishSegmentation::get_model_path()?;
        if !path.exists() {
            let bytes = FishSegmentation::download_model_bytes()?;
            match FishSegmentation::save_bytes_to_file(&bytes, &path) {
                Ok(byte_count) => {
                    if byte_count > 0 {
                        Ok(path)
                    } else {
                        Err(SegmentationError::CopyErr(byte_count))
                    }
                }
                Err(err) => Err(err)
            }
        } else {
            Ok(path)
        }
    }

    pub fn from_web() -> Result<FishSegmentation, SegmentationError> {
        let model_path = FishSegmentation::download_model()?;

        Ok(FishSegmentation::new(model_path))
    }

    pub fn new(model_path: PathBuf) -> FishSegmentation {
        FishSegmentation {
            model_path,
            model_set: false,
            model: None,
        }
    }

    fn create_model(&self) -> Result<Session, ort::Error> {
        Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&self.model_path)
    }

    pub fn load_model(&mut self) -> Result<(), SegmentationError> {
        if !self.model_set {
            match self.create_model() {
                Ok(model) => {
                    self.model = Some(model);
                    Ok(())
                }
                Err(error) => Err(SegmentationError::OrtErr(error))
            }?;

            self.model_set = true;
        }

        Ok(())
    }

    fn get_model(&self) -> Result<&Session, SegmentationError> {
        match &self.model {
            Some(model) => Ok(model),
            None => Err(SegmentationError::ModelLoadError)
        }
    }

    fn pad_img(&self, img: &Array3<u8>) -> Array3<u8> {
        let (height, width, _) = img.dim();

        let mut pad_img = if height < width {
            Array3::zeros((FishSegmentation::MIN_SIZE_TEST, FishSegmentation::MAX_SIZE_TEST, 3))
        }
        else {
            Array3::zeros((FishSegmentation::MAX_SIZE_TEST, FishSegmentation::MIN_SIZE_TEST, 3))
        };

        let mut slice = pad_img.slice_mut(s![..height, ..width, ..]);
        slice.assign(img);

        pad_img
    }

    fn resize_img(&self, img: &Array3<u8>) -> Result<Array3<u8>, SegmentationError> {
        let (height, width, _) = img.dim();
        println!("RUST: size: {}, {}", height, width);

        let size = FishSegmentation::MIN_SIZE_TEST as f32;
        let mut scale = size / min(height, width) as f32;
        println!("RUST: scale: {}", scale);

        let mut new_height_fl32: f32;
        let mut new_width_fl32: f32;
        if height < width {
            println!("RUST: height < width");

            new_height_fl32 = size;
            new_width_fl32 = scale * width as f32;

            println!("RUST: {}, {}", new_height_fl32, new_width_fl32);
        }
        else {
            println!("RUST: height >= width");
            new_height_fl32 = scale * height as f32;
            new_width_fl32 = size;
        }

        new_height_fl32 = new_height_fl32.round();
        new_width_fl32 = new_width_fl32.round();

        let max_size: usize = max(new_height_fl32 as usize, new_width_fl32 as usize);
        println!("RUST: max_size: {}", max_size);
        if  max_size > FishSegmentation::MAX_SIZE_TEST {
            println!("RUST: {} > {}", max_size, FishSegmentation::MAX_SIZE_TEST);
            scale = FishSegmentation::MAX_SIZE_TEST as f32 / max_size as f32;
            println!("RUST: scale: {}", scale);
            new_height_fl32 *= scale;
            new_width_fl32 *= scale;
            println!("RUST: {}, {}", new_height_fl32, new_width_fl32);
        }

        let new_height: usize = new_height_fl32 as usize;
        let new_width: usize = new_width_fl32 as usize;

        let conversion_result: Result<Mat, _> = img.try_into_cv();
        match conversion_result {
            Ok(mat) => {
                let mut resized_img_cv = Mat::default();
                match resize_def(&mat, &mut resized_img_cv, Size::new(new_width as i32, new_height as i32)) {
                    Ok(_) => {
                        let conversion_result: Result<Array3<u8>, _> = resized_img_cv.try_into_cv();
                        match conversion_result {
                            Ok(resized_img) => Ok(resized_img),
                            Err(_) => {
                                Err(SegmentationError::CVToNDArrayError)
                            }
                        }
                    },
                    Err(error) => Err(SegmentationError::OpenCVError(error))
                }
            }
            Err(_) => Err(SegmentationError::NDArrayToCVError)
        }
    }

    fn do_inference(&self, img: &Array3<f32>, model: &Session) ->
        Result<(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>), ort::Error> {
        let mut clone = img.clone();
        clone.swap_axes(2, 1);
        clone.swap_axes(1, 0);

        println!("RUST: clone: {}, {}, {}", clone.shape()[0], clone.shape()[1], clone.shape()[2]);

        println!("RUST: Before Run");
        let outputs = model.run(ort::inputs!["argument_1.1" => clone.view()]?)?;
        println!("RUST: After Run");

        // boxes=tensor18, classes=pred_classes, masks=5232, scores=2339, img_size=onnx::Split_174
        println!("RUST: Before parsing results");
        let boxes = outputs["tensor18"].try_extract_tensor::<f32>()?.t().into_owned();
        println!("RUST: Parsed tensor18");
        let masks = outputs["5232"].try_extract_tensor::<f32>()?.t().into_owned();
        println!("RUST: Parsed 5232");
        let scores = outputs["2339"].try_extract_tensor::<f32>()?.t().into_owned();
        println!("RUST: Parsed 2339");

        println!("RUST: Parsed results.");

        Ok((boxes, masks, scores))
    }

    fn do_paste_mask(&self, masks: &Array2<f32>, img_h: u32, img_w: u32) -> Result<Array2<f32>, SegmentationError> {
        let masks_unsqueezed = masks.clone().insert_axis(Axis(2));

        let conversion_result: Result<Mat, _> = masks_unsqueezed.try_into_cv();
        match conversion_result {
            Ok(masks_cv) => {
                let mut resized_cv = Mat::default();
                match resize_def(&masks_cv, &mut resized_cv, Size::new(img_w as i32, img_h as i32)) {
                    Ok(_) => {
                        let conversion_result: Result<Array3<f32>, _> = resized_cv.try_into_cv();
                        match conversion_result {
                            Ok(resized3) => {
                                let resized_mask = resized3.remove_axis(Axis(2));
                        
                                Ok(resized_mask)
                            },
                            Err(_) => Err(SegmentationError::CVToNDArrayError)
                        }
                    },
                    Err(error) => Err(SegmentationError::OpenCVError(error))
                }
            },
            Err(_) => Err(SegmentationError::NDArrayToCVError)
        }
    }

    fn bitmap_to_polygon(&self, bitmap: &Array2<u8>) -> Result<Vec<VectorOfPoint>, SegmentationError> {
        let bitmap3 = bitmap.clone().insert_axis(Axis(2));
        let conversion_result: Result<Mat, _> = bitmap3.try_into_cv();
        match conversion_result {
            Ok(bitmap_cv) => {
                let mut contours_cv = VectorOfVectorOfPoint::new();
                let mut hierarchy_cv = VectorOfVec4i::new();
                // cv2.RETR_CCOMP: retrieves all of the contours and organizes them
                //   into a two-level hierarchy. At the top level, there are external
                //   boundaries of the components. At the second level, there are
                //   boundaries of the holes. If there is another contour inside a hole
                //   of a connected component, it is still put at the top level.
                // cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
                match find_contours_with_hierarchy(&bitmap_cv, &mut contours_cv, &mut hierarchy_cv, RETR_CCOMP, CHAIN_APPROX_NONE, Point2i::new(0, 0)) {
                    Ok(_) => {
                        if hierarchy_cv.is_empty() {
                            Err(SegmentationError::FishNotFound)
                        }
                        else {
                            let mut vector = contours_cv.to_vec();
                            vector.sort_by(|a, b| b.len().cmp(&a.len()));

                            Ok(vector)
                        }
                    },
                    Err(error) => Err(SegmentationError::OpenCVError(error))
                }
            },
            Err(_) => Err(SegmentationError::NDArrayToCVError)
        }
    }

    fn rescale_polygon_to_src_size(&self, poly: &VectorOfPoint, start_x: u32, start_y: u32, width_scale: f32, height_scale: f32) -> VectorOfPoint {
        let res = VectorOfPoint::from_iter(poly
            .iter()
            .map(|point| Point2i::new(
                ((start_x as f32 + point.x as f32).ceil() * width_scale) as i32,
                ((start_y as f32 + point.y as f32).ceil() * height_scale) as i32
            ))
            .collect::<Vec<_>>());

        res 
    }

    fn convert_output_to_mask_and_polygons(
        &self,
        boxes: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        masks: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        scores: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        width_scale: f32, height_scale: f32,
        shape: (usize, usize, usize)) -> Result<Array2<u8>, SegmentationError> {

        let mut masks_clone = masks.clone();

        masks_clone.swap_axes(3, 2);
        masks_clone.swap_axes(2, 1);
        masks_clone.swap_axes(1, 0);
        masks_clone.swap_axes(1, 2);

        // let mut complete_mask = Array2::<u8>::zeros((shape.0, shape.1));
        match Mat::new_rows_cols_with_default(shape.0 as i32, shape.1 as i32, CV_8UC1, 0.into()) {
            Ok(mut complete_mask_cv) => {
                let mask_count = scores.len();

                for ind in 0..mask_count {
                    if scores[ind] <= FishSegmentation::SCORE_THRESHOLD {
                        continue;
                    }

                    let x1 = boxes[[0, ind]].ceil() as u32;
                    let y1 = boxes[[1, ind]].ceil() as u32;
                    let x2 = boxes[[2, ind]].floor() as u32;
                    let y2 = boxes[[3, ind]].floor() as u32;
                    let (mask_h, mask_w) = (y2 - y1 + 1, x2 - x1 + 1);

                    let mask = masks_clone.slice(s![ind, .., .., 0])
                        .mapv(|v| v.to_owned());

                    // Threshold the mask converting to uint8 casuse opencv diesn't allow other type!
                    let np_mask = self.do_paste_mask(&mask, mask_h, mask_w)?
                        .mapv(|v| if v > FishSegmentation::MASK_THRESHOLD {255 as u8} else {0});

                    // Find contours in the binary mask
                    match self.bitmap_to_polygon(&np_mask) {
                        Ok(contours) => {
                            // Ignore empty contpurs
                            if contours.is_empty() {
                                continue
                            }

                            // Ignore small artifacts
                            match contours.get(0) {
                                Some(poly) => {
                                    if poly.len() < 10 {
                                        continue
                                    }

                                    // Convert local polygon to src image
                                    let polygon_full = self.rescale_polygon_to_src_size(
                                        poly,
                                        x1, y1,
                                        width_scale, height_scale);

                                    let color = (ind + 1) as i32;
                                    match fill_poly(&mut complete_mask_cv, &polygon_full, (color, color, color).into(), LINE_8, 0, Point2i::new(0, 0)) {
                                        Ok(_) => Ok(()),
                                        Err(error) => Err(SegmentationError::OpenCVError(error))
                                    }
                                },
                                None => Err(SegmentationError::PolyNotFound)
                            }?;
                        },
                        Err(error) => {
                            match error {
                                SegmentationError::FishNotFound => Ok(()),
                                _ => Err(error)
                            }?
                        }
                    }
                }

                let conversion_result: Result<Array3<u8>, _> = (&complete_mask_cv).try_into_cv();
                match conversion_result {
                    Ok(complete_mask3) => {
                        let complete_mask = complete_mask3.remove_axis(Axis(2));
                        Ok(complete_mask)
                    }
                    Err(_) => {
                        Err(SegmentationError::CVToNDArrayError)
                    }
                }
            },
            Err(error) => Err(SegmentationError::OpenCVError(error))
        }
    }

    pub fn inference(&self, img: &Array3<u8>) -> Result<Array2<u8>, SegmentationError> {
        let model = self.get_model()?;
        let resized_img = self.resize_img(&img)?;
        let padded_img = self.pad_img(&resized_img).mapv(|v| v as f32);

        let (orig_height, orig_width, _) = img.dim();
        let (new_height, new_width, _) = resized_img.dim();

        println!("RUST: resized_size: {}, {}", new_height, new_width);

        let width_scale = orig_width as f32 / new_width as f32;
        let height_scale = orig_height as f32 / new_height as f32;

        match self.do_inference(&padded_img, model) {
            Ok(result) => {
                let (boxes, masks, scores) = result;
                let masks = self.convert_output_to_mask_and_polygons(&boxes, &masks, &scores, width_scale, height_scale, img.dim())?;

                Ok(masks)
            }
            Err(error) => Err(SegmentationError::OrtErr(error))
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray_npy::NpzReader;
    use ndarray_stats::DeviationExt;
    use std::fs::File;

    use super::*;

    #[test]
    fn inference() {
        let mut npz = NpzReader::new(File::open("data/fish_segmentation.npz").unwrap()).unwrap();
        let img8: Array3<u8> = npz.by_name("img8").unwrap();
        let truth: Array2<i32> = npz.by_name("segmentations").unwrap();
        
        let mut seg = FishSegmentation::from_web().unwrap();
        seg.load_model().unwrap();
        let segmentations = seg.inference(&img8).unwrap()
            .mapv(|v| v as i32);

        assert_eq!(segmentations.mean_abs_err(&truth).unwrap() < 2.0e-6, true);
    }
}