import matplotlib as mp
import tensorflow as tf
from matplotlib import pyplot as plt

# function for positioning plot window
def move_figure(f, x, y):
    backend = mp.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x,y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x,y))
    else:
        f.canvas.manager.window.move(x,y)

    
# for normalizing pixel values of input images
# and converting from rgb to grayscale for faster training process
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [100, 100])
    image = tf.image.rgb_to_grayscale(image)

    return image, label

# for resizing image and retaining colored image
def scale(image, label):
    image = tf.image.resize(image, [100, 100])

    return image, label