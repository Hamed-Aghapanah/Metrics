import MAL.phantom as ph
import matplotlib.pyplot as plt
import tensorflow as tf

rect = ph.Circle(n_classes=1)
rect_image = rect()
image = tf.convert_to_tensor(rect_image, dtype=tf.float32)
sobel = tf.image.sobel_edges(image)
sobel = sobel**2
sobel = tf.reduce_sum(sobel, axis=-1)
sobel = tf.sqrt(sobel)
sobel = tf.where(sobel > 0, 1, 0)
sobel = tf.nn.erosion2d(sobel, tf.ones((1, 1, 1), tf.int32), [1, 1, 1, 1], 'SAME', 'NHWC', [1, 1, 1, 1])
sobel = (sobel - tf.reduce_min(sobel))/(tf.reduce_max(sobel) - tf.reduce_min(sobel))
sobel = tf.cast(sobel, image.dtype)
print(sobel.dtype, image.dtype)
sobel = image*sobel
plt.imshow(sobel[0])
plt.show()