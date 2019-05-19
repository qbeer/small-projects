from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from preprocess import Preprocessor
import matplotlib.pyplot as plt


class TransferModel:
    def __init__(self, initial_image_path,
                 style_layers=['block1_conv1', 'block2_conv1',
                               'block3_conv1', 'block4_conv1',
                               'block5_conv1'],
                 content_layers=['block5_conv2']):
        self.style_layers = style_layers
        self.content_layers = content_layers
        """
            Loading the VGG19 model without its dense layers. Not trainable
            only the input image will change during optimization.
        """
        self.vgg = VGG19(weights='imagenet', include_top=False)
        self.vgg.trainable = False

        self.model_style_layers = [self.vgg.get_layer(
            name).output for name in self.style_layers]
        self.model_content_layers = [self.vgg.get_layer(
            name).output for name in self.content_layers]
        """
            The model output is the combination from layers for style and content
            reconstruction.
        """
        self.model = Model(
            self.vgg.input, self.model_style_layers + self.model_content_layers)

        self.prep = Preprocessor()
        self.initial_image = self.prep.get_preprocessed_input(
            initial_image_path)
        self.initial_image = tf.Variable(self.initial_image, dtype=tf.float32)

    def _content_loss(self, base_content_features, generated_content_features):
        return 0.5*tf.reduce_sum(tf.square(generated_content_features - base_content_features))

    def _gram_matrix(self, feature):
        return tf.matmul(feature, feature, transpose_b=True)

    def _style_loss(self, style_image_features, generated_image_features):
        style_gram, generated_gram = self._gram_matrix(
            style_image_features), self._gram_matrix(generated_image_features)
        _, width, _, channels = tf.shape(style_gram)
        return tf.reduce_sum(tf.square(generated_gram - style_gram)) \
            / tf.cast(4 * channels**2 * width**2, dtype=tf.float32)

    def _get_style_features(self, style_image):
        features = self.model(style_image)
        features = [style_layer
                    for style_layer in features[:len(self.style_layers)]]
        return features

    def _get_content_features(self, content_image):
        features = self.model(content_image)
        features = [content_layer
                    for content_layer in features[len(self.style_layers):]]
        return features

    def compute_loss(self, loss_weights):
        style_weight, content_weight = loss_weights

        model_outputs = self.model(self.initial_image)
        style_outputs = model_outputs[:len(self.style_layers)]
        content_outputs = model_outputs[len(self.style_layers):]

        style_score = 0
        content_score = 0

        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        for target_style, comb_style in zip(self.style_features, style_outputs):
            style_score += weight_per_style_layer * \
                self._style_loss(target_style, comb_style)

        weight_per_content_layer = 1.0 / float(len(self.content_layers))
        for target_content, comb_content in zip(self.content_features, content_outputs):
            content_score += weight_per_content_layer * \
                self._content_loss(target_content, comb_content)

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, loss_weights):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(loss_weights)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, self.initial_image), all_loss

    def style_transfer(self, content_path,
                       style_path, max_iter=1000,
                       content_weight=1e2, style_weight=1e-2):
        """
            Freezing the model!
        """
        for layer in self.model.layers:
            layer.trainable = False

        style_image = self.prep.get_preprocessed_input(style_path)
        self.style_features = self._get_style_features(style_image)

        content_image = self.prep.get_preprocessed_input(content_path)
        self.content_features = self._get_content_features(content_image)

        print("Style : ", np.min(style_image), np.max(style_image))
        print("Content : ", np.min(content_image), np.max(content_image))

        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
        loss_weights = (style_weight, content_weight)

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        for i in range(max_iter):
            grads, all_loss = self.compute_grads(loss_weights)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, self.initial_image)])
            clipped = tf.clip_by_value(self.initial_image, min_vals, max_vals)
            self.initial_image.assign(clipped)
            if i % 100 == 0:
                print(
                    "Iter %d : loss %.2f, style_score %.2f, content_score %.2f" % (
                        i + 1, loss, style_score, content_score))
                print("Ndim : ", self.initial_image.numpy().ndim)
                img = self.initial_image.numpy()

                img[:, :, 0] += 103.939
                img[:, :, 1] += 116.779
                img[:, :, 2] += 123.68

                img = img[:, :, ::-1]
                img = np.clip(img, 0, 255).astype(int)

                plt.imshow(img)
                plt.show()
