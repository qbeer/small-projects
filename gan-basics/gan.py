# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Reshape,\
    Conv2D, Conv2DTranspose, Dropout, Flatten
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

print('TensorFlow version %s' % tf.__version__)
print('TensorFlow Probability version %s' % tfp.__version__)

#######################################################################################################################################
# Ref: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb ###
#######################################################################################################################################


class GAN:
    def __init__(self,
                 input_shape=(28, 28, 1),
                 encoder=None,
                 decoder=None,
                 number_of_examples=16):
        self.input_shape = input_shape
        self.generator = self._generator()
        self.discriminator = self._discriminator()

        self.generator_noise_vector = tfd.Normal(loc=[0.] * 100,
                                                 scale=[1.] * 100).sample(
                                                     [number_of_examples])

    def _generator(self):
        model = Sequential()

        model.add(Dense(7 * 7 * 32, input_shape=(100, )))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Reshape((7, 7, 32)))

        model.add(Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        # 14, 14, 32

        model.add(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        # 28, 28, 16

        model.add(
            Conv2DTranspose(1, (2, 2),
                            strides=(1, 1),
                            padding='same',
                            activation='sigmoid'))
        # 28, 28, 1
        return model

    def _discriminator(self):
        model = Sequential()
        model.add(
            Conv2D(64, (2, 2),
                   strides=(2, 2),
                   padding='same',
                   input_shape=self.input_shape))
        model.add(ReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (2, 2), strides=(2, 2), padding='same'))
        model.add(ReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        return model

    def _generator_loss(self, generated):
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated),
                                               generated)

    def _discriminator_loss(self, real, generated):
        real_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real), logits=real)
        generated_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(generated), logits=generated)
        total_loss = real_loss + generated_loss
        return total_loss

    def _train_step(self, images, batch_size, noise_dim=100):
        noise_vec = tfd.Normal(loc=[0.] * noise_dim,
                               scale=[1.] * noise_dim).sample([batch_size])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = self.generator(noise_vec)

            real_out = self.discriminator(images)
            gen_out = self.discriminator(generated)

            gen_loss = self._generator_loss(gen_out)
            disc_loss = self._discriminator_loss(real_out, gen_out)

        grads_of_gen = gen_tape.gradient(gen_loss, self.generator.variables)
        grads_of_disc = disc_tape.gradient(disc_loss,
                                           self.discriminator.variables)

        tf.train.AdamOptimizer(1e-4).apply_gradients(
            zip(grads_of_gen, self.generator.variables))
        tf.train.AdamOptimizer(1e-4).apply_gradients(
            zip(grads_of_disc, self.discriminator.variables))

    def train(self, dataset, batch_size, epochs=50):
        for epoch in range(epochs):
            for images in dataset:
                self._train_step(images, batch_size)

            self._generate_and_save_images(epoch + 1)

    def _generate_and_save_images(self, current_epoch):
        generated = self._generator(self.generator_noise_vector)
        fig, axes = plt.subplots(4,
                                 4,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(7, 7))
        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(generated[ind], vmin=0, vmax=1)
        fig.suptitle('Generated images')
        fig.tight_layout()
        plt.savefig('generated_after_epoch_%d.png' % current_epoch)
