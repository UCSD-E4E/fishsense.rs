use std::cmp::{max, min};
use std::fs::{File, create_dir};
use std::io::{Cursor, copy};
use std::path::PathBuf;

use app_dirs2::{AppDataType, AppInfo, app_root};
use bytes::Bytes;
use image::RgbImage;
use image::imageops::{resize, FilterType};
use ndarray::{s, Array2, Array3, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::Session;
use reqwest::blocking::get;

#[derive(Debug)]
pub enum SegmentationError {
    AppDirsError(app_dirs2::AppDirsError),
    ArrayShapeError(ndarray::ShapeError),
    CopyErr(u64),
    DownloadError(reqwest::Error),
    GenericError,
    IOError(std::io::Error),
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
    const NMS_THRESHOLD: f32 = 0.9;

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

    fn convert_output_to_mask_and_polygons(
        &self,
        boxes: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        masks: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        scores: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        shape: (usize, usize, usize)) -> Array2<u8> {

        let complete_mask = Array2::<u8>::zeros((shape.0, shape.1));
        let mask_count = masks.shape().len();

        for ind in 0..mask_count {
            if scores[ind] <= FishSegmentation::SCORE_THRESHOLD {
                continue;
            }

            // let x1 = boxes[[0, ind]].round() as u32;
            // let y1 = boxes[[1, ind]].round() as u32;
            // let x2 = boxes[[2, ind]].round() as u32;
            // let y2 = boxes[[3, ind]].round() as u32;

            // let (mask_h, mask_w) = (y2 - y1, x2 - x1);
        }

        complete_mask
    }

    pub fn inference(&self, img: Array3<u8>) -> Result<Array2<u8>, SegmentationError> {
        let model = self.get_model()?;
        let resized_img = self.resize_img(&img)?;

        let (orig_height, orig_width, _) = img.dim();
        let (new_height, new_width, _) = resized_img.dim();

        let height_scale = orig_height as f32 / new_height as f32;
        let width_scale = orig_width as f32 / new_width as f32;

        match self.do_inference(&resized_img, model) {
            Ok(result) => {
                let (boxes, masks, scores) = result;
                let masks = self.convert_output_to_mask_and_polygons(&boxes, &masks, &scores, img.dim());

                Ok(masks)
            }
            Err(error) => Err(SegmentationError::OrtErr(error))
        }


        //model.inputs.first()

        // ort_inputs = {
        //     self.ort_session.get_inputs()[0]
        //     .name: resized_img.astype("float32")
        //     .transpose(2, 0, 1)
        // }
        // ort_outs = self.ort_session.run(None, ort_inputs)

        //let outputs = model.run(ort::inputs!["argument_1.1" => ])

        // argument_1.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference() {
        let rust_img = image::io::Reader::open("./data/img8.png").unwrap().decode().unwrap().as_mut_rgb8().unwrap().clone();
        let img = Array3::from_shape_vec((rust_img.height() as usize, rust_img.width() as usize, 3), rust_img.as_raw().clone()).unwrap();
        
        let mut seg = FishSegmentation::from_web().unwrap();
        seg.load_model().unwrap();
        let res = seg.inference(img).unwrap();
    }
}