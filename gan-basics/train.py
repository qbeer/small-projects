# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tf.enable_eager_execution()

from gan import GAN

(train_images,
 train_labels), (test_images,
                 test_labels) = tf.keras.datasets.mnist.load_data()

train_images, train_labels = test_images, test_labels
train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                    1).astype('float32')
train_images = train_images / 255.  # Normalize the images to [0, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

gan = GAN()

gan.train(train_dataset, batch_size=BATCH_SIZE, epochs=100)

from PIL import Image
import glob

# Create the frames
frames = []
imgs = glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('mnist.gif',
               format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=150,
               loop=0)