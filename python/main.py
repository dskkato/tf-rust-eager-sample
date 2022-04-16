import tensorflow as tf

# 画像を読み込む
fname = "sample/macaque.jpg"
buf = tf.io.read_file(fname)
img = tf.image.decode_image(buf)

# floatに変換して[0,1]に正規化
img = tf.cast(img, tf.float32)
img = img / 255.0

# バッチサイズ1に変換
batch = tf.expand_dims(img, 0)

# リサイズする
resized = tf.image.resize(batch, [224, 224], "bilinear", antialias=True)

# 1ピクセル目の値を確認する。
print(f"{resized[0, 0, 0, :3]}")
# [0.29298395 0.35878524 0.4291904 ]
