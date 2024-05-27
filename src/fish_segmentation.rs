use crate::cv_conversion;

use std::cmp::{max, min};
use std::fs::{File, create_dir};
use std::io::{Cursor, copy};
use std::path::PathBuf;

use app_dirs2::{AppDataType, AppInfo, app_root};
use bytes::Bytes;
use image::RgbImage;
use image::imageops::{resize, FilterType};
use ndarray::{array, s, stack, Array, Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use ndarray_npy::NpzWriter;
use opencv::core::{Mat, MatTraitConstManual, Point2i, CV_8UC1};
use opencv::imgproc::{fill_poly, find_contours_with_hierarchy, CHAIN_APPROX_NONE, LINE_8, RETR_CCOMP};
use opencv::types::{VectorOfPoint, VectorOfVec4i, VectorOfVectorOfPoint};
use ort::Session;
use reqwest::blocking::get;

use self::cv_conversion::as_mat_u8c1_mut;


fn write(arr: Array3<f32>) {
    let mut npz = NpzWriter::new(File::create("../outputs/rust.npz").unwrap());
    npz.add_array("arr", &arr).unwrap();
    npz.finish().unwrap();
}



#[derive(Debug)]
pub enum SegmentationError {
    AppDirsError(app_dirs2::AppDirsError),
    ArrayShapeError(ndarray::ShapeError),
    CopyErr(u64),
    DownloadError(reqwest::Error),
    FishNotFound,
    GenericError,
    IOError(std::io::Error),
    OpenCVError(opencv::Error),
    OrtErr(ort::Error),
}

pub struct FishSegmentation {
    model_path: PathBuf,
    model_set: bool,
    model: Option<Session>,
}

impl FishSegmentation {
    const MIN_SIZE_TEST: usize = 800;
    const MAX_SIZE_TEST: usize = 1333;

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
            None => Err(SegmentationError::GenericError)
        }
    }

    fn resize_img(&self, img: &Array3<u8>) -> Result<Array3<f32>, SegmentationError> {
        let (height, width, _) = img.dim();

        let size = FishSegmentation::MIN_SIZE_TEST as f32 * 1.0;
        let mut scale = size / min(height, width) as f32;

        let mut new_height_fl32: f32;
        let mut new_width_fl32: f32;
        if height < width {
            new_height_fl32 = size;
            new_width_fl32 = scale * width as f32;
        }
        else {
            new_height_fl32 = scale * height as f32;
            new_width_fl32 = size;
        }

        let max_size: usize = max(new_height_fl32 as usize, new_width_fl32 as usize);
        if  max_size > FishSegmentation::MAX_SIZE_TEST {
            scale = FishSegmentation::MAX_SIZE_TEST as f32 / max_size as f32;
            new_height_fl32 *= scale;
            new_width_fl32 *= scale;
        }

        new_height_fl32 += 0.5;
        new_width_fl32 += 0.5;
        let new_height: usize = new_height_fl32 as usize;
        let new_width: usize = new_width_fl32 as usize;

        match RgbImage::from_raw(width as u32, height as u32, img.clone().into_raw_vec()) {
            Some(img) => {
                let resized_img = resize(&img, new_width as u32, new_height as u32, FilterType::Lanczos3);
                match Array3::from_shape_vec((new_height, new_width, 3), resized_img.as_raw().clone()) {
                    Ok(resized_img) => Ok(resized_img.mapv(|x| f32::from(x))),
                    Err(error) => Err(SegmentationError::ArrayShapeError(error))
                }
            },
            None => Err(SegmentationError::GenericError)
        }
    }

    fn do_inference(&self, img: &Array3<f32>, model: &Session) ->
        Result<(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>), ort::Error> {
        let mut clone = img.clone();
        clone.swap_axes(2, 1);
        clone.swap_axes(1, 0);

        let outputs = model.run(ort::inputs!["argument_1.1" => clone.view()]?)?;

        // boxes=tensor18, classes=pred_classes, masks=5232, scores=2339, img_size=onnx::Split_174
        let boxes = outputs["tensor18"].try_extract_tensor::<f32>()?.t().into_owned();
        let masks = outputs["5232"].try_extract_tensor::<f32>()?.t().into_owned();
        let scores = outputs["2339"].try_extract_tensor::<f32>()?.t().into_owned();

        Ok((boxes, masks, scores))
    }

    fn grid_sample(&self, input: &ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 4]>>, grid: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>) -> Array2::<f32> {
        let (h_in, w_in, _, _) = input.dim();
        let (_, h_out, w_out, _) = grid.dim();

        let mut output = Array2::<f32>::zeros((h_out, w_out));
        for h in 0..h_out {
            for w in 0..w_out {
                let x = (grid[[0, h, w, 0]] + 1.0) / 2.0 / w_in as f32;
                let y = (grid[[0, h, w, 1]] + 1.0) / 2.0 / h_in as f32;
                let x0 = x.floor();
                let x1 = x.ceil();
                let y0 = y.floor();
                let y1 = y.ceil();

                // See: https://en.wikipedia.org/wiki/Bilinear_interpolation
                let q00 = input[[y0 as usize, x0 as usize, 0, 0]];
                let q01 = input[[y1 as usize, x0 as usize, 0, 0]];
                let q10 = input[[y0 as usize, x1 as usize, 0, 0]];
                let q11 = input[[y1 as usize, x1 as usize, 0, 0]];

                let frac = 1.0 / ((x1 - x0) * (y1 - y0));
                let mat = array![[(x1 - x), (x - x0)]].dot(&array![[q00, q01], [q10, q11]]).dot(&array![[y1 - y], [y - y0]]);

                output[[h, w]] = frac * mat[[0, 0]];
            }
        }

        output
    }

    fn do_paste_mask(&self, masks: &ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 4]>>, img_h: u32, img_w: u32) -> Result<Array2::<f32>, SegmentationError> {
        let x0_int: f32 = 0.0;
        let y0_int: f32 = 0.0;
        let x1_int = img_w as f32;
        let y1_int = img_h as f32;

        let x0 = array![[0.0]];
        let y0 = array![[0.0]];
        let x1 = array![[img_w as f32]];
        let y1 = array![[img_h as f32]];

        let mut img_y = (Array::range(y0_int, y1_int, 1.0) + 0.5).insert_axis(Axis(0));
        let mut img_x = (Array::range(x0_int, x1_int, 1.0) + 0.5).insert_axis(Axis(0));
        img_y = (img_y - &y0) / (&y1 - &y0) * 2.0 - 1.0;
        img_x = (img_x - &x0) / (&x1 - &x0) * 2.0 - 1.0;

        let img_y_len = img_y.len();
        let img_x_len = img_x.len();

        let gy_vec = img_y.into_raw_vec().iter().flat_map(|&f| std::iter::repeat(f.clone()).take(img_x_len)).collect::<Vec<f32>>();
        match Array3::from_shape_vec((1, img_y_len, img_x_len), gy_vec) {
            Ok(gy) => {
                let gx_vec = img_x.into_raw_vec().iter().map(|&f| f.clone()).cycle().take(img_x_len * img_y_len).collect::<Vec<f32>>();
                match Array3::from_shape_vec((1, img_y_len, img_x_len), gx_vec) {
                    Ok(gx) => {
                        let grid = stack![Axis(3), gx, gy];


                        let resized_mask = self.grid_sample(&masks, &grid);
                        Ok(resized_mask)
                    },
                    Err(error) => Err(SegmentationError::ArrayShapeError(error))
                }
            },
            Err(error) => Err(SegmentationError::ArrayShapeError(error))
        }
    }

    fn bitmap_to_polygon(&self, bitmap: &Array2<u8>) -> Result<VectorOfVectorOfPoint, SegmentationError> {
        match as_mat_u8c1_mut(bitmap) {
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
                            Ok(contours_cv)
                        }
                    },
                    Err(error) => Err(SegmentationError::OpenCVError(error))
                }
            },
            Err(error) => Err(SegmentationError::OpenCVError(error))
        }
    }

    fn rescale_polygon_to_src_size(&self, poly: VectorOfPoint, start_x: u32, start_y: u32, width_scale: f32, height_scale: f32) -> VectorOfPoint {
        let res = VectorOfPoint::from_iter(poly
            .iter()
            .map(|point| Point2i::new(
                ((start_x as f32 + point.x as f32) * width_scale) as i32,
                ((start_y as f32 + point.y as f32) * height_scale) as i32
            ))
            .collect::<Vec<_>>());

        println!("{}, {}", res.get(0).unwrap().x, res.get(0).unwrap().y);
        res 
    }

    fn as_u8c1_mat_array2(&self, mat: &Mat, shape: (usize, usize)) -> Result<Array2<u8>, SegmentationError> {
        let vec_result: Result<Vec<Vec<u8>>, _> = mat.to_vec_2d();
        match vec_result {
            Ok(vec) => {
                match Array2::from_shape_vec(shape, vec.iter().flatten().map(|v| v.clone()).collect()) {
                    Ok(arr) => Ok(arr),
                    Err(error) => Err(SegmentationError::ArrayShapeError(error))
                }
            },
            Err(err) => Err(SegmentationError::OpenCVError(err))
        }
    }

    fn convert_output_to_mask_and_polygons(
        &self,
        boxes: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        masks: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        scores: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        width_scale: f32, height_scale: f32,
        shape: (usize, usize, usize)) -> Result<Array2<u8>, SegmentationError> {

        // let mut complete_mask = Array2::<u8>::zeros((shape.0, shape.1));
        match Mat::new_rows_cols_with_default(shape.0 as i32, shape.1 as i32, CV_8UC1, 0.into()) {
            Ok(mut complete_mask_cv) => {
                let mask_count = scores.len();

                for ind in 0..mask_count {
                    if scores[ind] <= FishSegmentation::SCORE_THRESHOLD {
                        println!("scores below thresh");
                        continue;
                    }

                    let x1 = boxes[[0, ind]].ceil() as u32;
                    let y1 = boxes[[1, ind]].ceil() as u32;
                    let x2 = boxes[[2, ind]].floor() as u32;
                    let y2 = boxes[[3, ind]].floor() as u32;
                    let (mask_h, mask_w) = (y2 - y1, x2 - x1);

                    let mask = masks.slice(s![.., .., .., ind]).insert_axis(Axis(3));
                    // Threshold the mask converting to uint8 casuse opencv diesn't allow other type!
                    let np_mask = self.do_paste_mask(&mask, mask_h, mask_w)?
                        .mapv(|v| if v > FishSegmentation::MASK_THRESHOLD {255 as u8} else {0});

                    // Find contours in the binary mask
                    let contours = self.bitmap_to_polygon(&np_mask)?;

                    // Ignore empty contpurs
                    if contours.is_empty() {
                        println!("contours empty");
                        continue
                    }

                    // Ignore small artifacts
                    match contours.get(0) {
                        Ok(poly) => {
                            if poly.len() < 10 {
                                println!("contours small");
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
                        Err(error) => Err(SegmentationError::OpenCVError(error))
                    }?;
                }

                
                Ok(self.as_u8c1_mat_array2(&complete_mask_cv, (shape.0, shape.1))?)
            },
            Err(error) => Err(SegmentationError::OpenCVError(error))
        }
    }

    pub fn inference(&self, img: &Array3<u8>) -> Result<Array2<u8>, SegmentationError> {
        let model = self.get_model()?;
        let resized_img = self.resize_img(&img)?;

        // println!("{}", resized_img);
        write(resized_img.clone());

        let (orig_height, orig_width, _) = img.dim();
        let (new_height, new_width, _) = resized_img.dim();

        let width_scale = orig_width as f32 / new_width as f32;
        let height_scale = orig_height as f32 / new_height as f32;

        match self.do_inference(&resized_img, model) {
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
    use super::*;

    #[test]
    fn inference() {
        let rust_img = image::io::Reader::open("./data/img8.png").unwrap().decode().unwrap().as_rgb8().unwrap().clone();
        let img_rgb = Array3::from_shape_vec((rust_img.height() as usize, rust_img.width() as usize, 3), rust_img.as_raw().clone()).unwrap();

        let (height, width, _) = img_rgb.dim();
        let mut img_bgr = Array3::<u8>::zeros(img_rgb.dim());
        
        for y in 0..height {
            for x in 0..width {
                img_bgr[[y, x, 0]] = img_rgb[[y, x, 2]];
                img_bgr[[y, x, 1]] = img_rgb[[y, x, 1]];
                img_bgr[[y, x, 2]] = img_rgb[[y, x, 0]];
            }
        }

        let rust_segmentations = image::io::Reader::open("./data/segmentations.png").unwrap().decode().unwrap().as_luma8().unwrap().clone();
        let truth = Array2::from_shape_vec((rust_segmentations.height() as usize, rust_segmentations.width() as usize), rust_segmentations.as_raw().clone()).unwrap()
            .mapv(|v| v as i32);
        
        let mut seg = FishSegmentation::from_web().unwrap();
        seg.load_model().unwrap();
        let segmentations = seg.inference(&img_bgr).unwrap()
            .mapv(|v| v as i32);

        // println!("truth: {}, seg: {}", truth.into_raw_vec().iter().max().unwrap(), segmentations.into_raw_vec().iter().max().unwrap());
        // let res = truth - segmentations;
        println!("{}, {}", truth.sum(), segmentations.sum());
        // println!("{}", res.sum());
        // assert_eq!(res.sum(), 0);
    }
}