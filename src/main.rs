use std::error::Error;

use tensorflow as tf;
use tf::eager::{self, raw_ops, ReadonlyTensor, TensorHandle, ToTensorHandle};
use tf::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    // eager API実行のコンテキストを作る。GPUの使用や、デバイスを指定することができる。
    let opts = eager::ContextOptions::new();
    let ctx = eager::Context::new(opts)?;

    // eager APIを使って画像を読み込み
    let fname: TensorHandle = "sample/macaque.jpg".to_handle(&ctx)?;
    let buf: TensorHandle = raw_ops::read_file(&ctx, &fname)?;
    let img: TensorHandle = raw_ops::decode_image(&ctx, &buf)?;

    // 画像サイズを確認する。
    let height = img.dim(0)?;
    let width = img.dim(1)?;
    assert!(height == 400);
    assert!(width == 600);

    // 後で[0, 1]に正規化するために、floatにキャストしておく
    let cast2float = raw_ops::Cast::new().DstT(tf::DataType::Float);
    let img = cast2float.call(&ctx, &img)?;
    assert!(img.data_type() == tf::DataType::Float);

    // [0, 1]に正規化する。255.0とすると、型の不一致でエラーになる。
    let img = raw_ops::div(&ctx, &img, &255.0f32)?;

    // HWC -> NHWC に変換する
    let batch = raw_ops::expand_dims(&ctx, &img, &0)?;

    // [224, 224, 3]にリサイズする。
    // ここではantialiasを有効にするために、v2のAPIを使う。
    let resize_bilinear = raw_ops::ScaleAndTranslate::new()
        .kernel_type("triangle") // bilinearのオプションに相当
        .antialias(true);
    let scale = [224.0 / height as f32, 224.0 / width as f32];
    let resized = resize_bilinear.call(&ctx, &batch, &[224, 224], &scale, &[0f32, 0f32])?;

    // Tensorの中身にアクセスできるように、TensorHandleからTensorに戻す
    // 今の実装では、ReadonlyTensorを経由してTensorに戻す必要がある。
    let readonly_t: ReadonlyTensor<f32> = resized.resolve()?;
    let t: Tensor<f32> = unsafe { readonly_t.into_tensor() };

    // resize後の1つ目のピクセルについて、
    // Pythonで計算した結果と比較する
    assert!((t[0] - 0.29298395).abs() < 1e-5);
    assert!((t[1] - 0.35878524).abs() < 1e-5);
    assert!((t[2] - 0.42919040).abs() < 1e-5);

    Ok(())
}
