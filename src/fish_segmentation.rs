use std::cmp::{max, min};
use std::fs::{File, create_dir};
use std::io::{Cursor, copy};
use std::path::PathBuf;

use app_dirs2::{AppDataType, AppInfo, app_root};
use bytes::Bytes;
use ndarray::{Array, Dim};
use ort::Session;
use reqwest::blocking::get;

#[derive(Debug)]
pub enum SegmentationError {
    AppDirsError(app_dirs2::AppDirsError),
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

    fn resize_img(&self, img: &Array<f32, Dim<[usize; 3]>>) -> Array<f32, Dim<[usize; 3]>> {
        let height = img.dim().0;
        let width = img.dim().1;

        let size = FishSegmentation::MIN_SIZE_TEST as f32 * 1.0;
        let mut scale = size / min(height, width) as f32;

        let mut new_height_fl32: f32 = 0.0;
        let mut new_width_fl32: f32 = 0.0;
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

        img.clone()
    }

    fn do_inference(&self, img: &Array<f32, Dim<[usize; 3]>>, model: &Session) -> Result<(), ort::Error> {
        let outputs = model.run(ort::inputs!["argument_1.1" => img.view()]?)?;
        let predictions = outputs["output0"].try_extract_tensor::<f32>()?;

        Ok(())
    }

    pub fn inference(&self, img: Array<f32, Dim<[usize; 3]>>) -> Result<(), SegmentationError> {
        let model = self.get_model()?;
        let resized_img = self.resize_img(&img);
        
        // match self.do_inference(&resized_img, model) {
        //     Ok(_) => Ok(()),
        //     Err(error) => Err(SegmentationError::OrtErr(error))
        // }

        //model.inputs.first()

        // ort_inputs = {
        //     self.ort_session.get_inputs()[0]
        //     .name: resized_img.astype("float32")
        //     .transpose(2, 0, 1)
        // }
        // ort_outs = self.ort_session.run(None, ort_inputs)

        //let outputs = model.run(ort::inputs!["argument_1.1" => ])

        // argument_1.1

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference() {
        let img = Array::zeros((640, 800, 3));

        let mut seg = FishSegmentation::from_web().unwrap();
        seg.load_model().unwrap();
        let res = seg.inference(img).unwrap();
    }
}