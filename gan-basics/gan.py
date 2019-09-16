# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape,\
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

        model.add(Dense(7 * 7 * 128, input_shape=(100, )))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 128)))

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        # 14, 14, 32

        model.add(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        # 28, 28, 32

        model.add(Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='same'))
        # 28, 28, 1
        return model

    def _discriminator(self):
        model = Sequential()
        model.add(
            Conv2D(64, (3, 3),
                   strides=(2, 2),
                   padding='same',
                   input_shape=self.input_shape))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    def _generator_loss(self, generated):
        return -tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated),
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

        tf.train.AdamOptimizer(1e-3).apply_gradients(
            zip(grads_of_gen, self.generator.variables))

        capped_disc_grad = [
            tf.clip_by_value(grad, -1., 1.) for grad in grads_of_disc
        ]
        tf.train.AdamOptimizer(1e-3).apply_gradients(
            zip(capped_disc_grad, self.discriminator.variables))

    def train(self, dataset, batch_size, epochs=50):

        for epoch in range(epochs):
            print('Epoch : %d' % (epoch + 1))
            for images in dataset:
                self._train_step(images, batch_size)

            self._generate_and_save_images(epoch + 1)

    def _generate_and_save_images(self, current_epoch):
        generated = self.generator(self.generator_noise_vector)
        fig, axes = plt.subplots(4,
                                 4,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(10, 10))
        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(generated[ind].numpy().reshape(28, 28), vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.savefig('images/test_generated_after_epoch_%d.png' % current_epoch)
        plt.close(fig)
