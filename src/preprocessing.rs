use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, ImageBuffer, RgbImage};
use ndarray::{concatenate, s, stack, Array, Array3, Array4, Axis};
use std::path::Path;

const MASK_WIDTH: u32 = 804;
const MASK_HEIGHT: u32 = 32;

pub fn preprocess_images_from_path<P: AsRef<Path>>(image_paths: &[P]) -> Result<Array4<f32>> {
    let mut image_data = Vec::new();
    for path in image_paths {
        let img = image::open(path)?;
        image_data.push(img);
    }
    process_images(&image_data)
}

pub fn preprocess_images_from_memory(image_data: &[&[u8]]) -> Result<Array4<f32>> {
    let mut image = Vec::new();
    for data in image_data {
        let img = image::load_from_memory(data)?;
        image.push(img);
    }
    process_images(&image)
}

fn keepratio_resize(img: &RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let cur_ratio = width as f32 / height as f32;
    let target_ratio = MASK_WIDTH as f32 / MASK_HEIGHT as f32;

    let (cur_target_width, cur_target_height) = if cur_ratio > target_ratio {
        (MASK_WIDTH, MASK_HEIGHT)
    } else {
        ((MASK_HEIGHT as f32 * cur_ratio).round() as u32, MASK_HEIGHT)
    };

    let resized_img = image::imageops::resize(
        img,
        cur_target_width,
        cur_target_height,
        FilterType::Nearest,
    );

    let mut mask = ImageBuffer::from_pixel(MASK_WIDTH, MASK_HEIGHT, image::Rgb([0, 0, 0]));

    for y in 0..resized_img.height() {
        for x in 0..resized_img.width() {
            let pixel = resized_img.get_pixel(x, y);
            mask.put_pixel(x, y, *pixel);
        }
    }

    mask
}

fn process_chunks(img_array: Array3<u8>) -> Array4<f32> {
    let img_width = img_array.shape()[1];
    let mut chunk_imgs = Vec::new();
    for i in 0..3 {
        let left = ((300 - 48) * i) as usize;
        let right = (left + 300).min(img_width as usize);
        let chunk = img_array.slice(s![.., left..right, ..]).to_owned();
        chunk_imgs.push(chunk);
    }

    let merge_img = stack(
        Axis(0),
        &chunk_imgs.iter().map(|x| x.view()).collect::<Vec<_>>(),
    )
    .unwrap();

    merge_img.mapv(|x| x as f32 / 255.0)
}

fn process_images(image_data: &[DynamicImage]) -> Result<Array4<f32>> {
    let mut batch_imgs = Vec::new();
    for data in image_data {
        let img = data.to_rgb8();
        let img = keepratio_resize(&img);
        let (img_width, img_height) = img.dimensions();
        let img_data = img.into_raw();
        let img_array =
            Array::from_shape_vec((img_height as usize, img_width as usize, 3), img_data)?;
        let processed_img = process_chunks(img_array);
        batch_imgs.push(processed_img);
    }
    let batch_data = concatenate(
        Axis(0),
        &batch_imgs.iter().map(|x| x.view()).collect::<Vec<_>>(),
    )?;

    Ok(batch_data.permuted_axes([0, 3, 1, 2]))
}
