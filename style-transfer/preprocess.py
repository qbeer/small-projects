from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input


class Preprocessor:
    def load(self, path):
        img = Image.open(path)
        width, height = 400, 400
        img = img.resize((width, height), resample=Image.BICUBIC)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def get_preprocessed_input(self, image_path):
        img = self.load(image_path)
        img = preprocess_input(img)
        return img
