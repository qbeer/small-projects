import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import os
from model import TransferModel
import numpy as np

model = TransferModel(os.getcwd() + "/hun_parliament.jpg")

model.style_transfer(os.getcwd() + "/hun_parliament.jpg", os.getcwd() + "/van_gogh_the_starry_night.jpg")

img = model.initial_image.numpy()[0]

print(np.min(img), np.max(img))

img[:,:,0] += 103.939
img[:,:,1] += 116.779
img[:,:,2] += 123.68

img = img[:,:,::-1]
img = np.clip(img, 0, 255).astype(int)

print(np.min(img), np.max(img))

plt.imshow(img)
plt.show()
