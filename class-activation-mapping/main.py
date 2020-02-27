import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

imagenette = tfds.load('imagenette/320px', split=tfds.Split.TRAIN)
imagenette_validation = tfds.load('imagenette/320px', split=tfds.Split.VALIDATION)

@tf.function
def apply_preproc(example):
    image, label = example['image'], example['label']
    image = tf.image.resize(image, [320, 320])
    image = tf.keras.applications.vgg19.preprocess_input(image)
    label = tf.one_hot(label, depth=10)
    return image, label

imagenette = imagenette.shuffle(buffer_size=15000).map(apply_preproc).batch(16)

imagenette_validation = imagenette_validation.shuffle(buffer_size=1500).map(apply_preproc).batch(8)

def create_model():
    input_image = tf.keras.layers.Input(shape=(320, 320, 3))
    vgg19 = tf.keras.applications.VGG19(weights='imagenet', input_shape=(320, 320, 3),
                                        include_top=False, pooling=None)
    
    x = vgg19(input_image)
    
    avg_pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(10, activation='softmax', use_bias=False)(avg_pooled)

    model = tf.keras.Model(inputs=input_image, outputs=[output, avg_pooled, x])

    return model

optimizer = tf.keras.optimizers.Adam(lr=1e-5)
cce = tf.keras.losses.CategoricalCrossentropy()

model = create_model()
model.load_weights('weights.h5')

acc = tf.keras.metrics.CategoricalAccuracy()

for images, labels in imagenette_validation:
    pred_labels, _, _ = model(images, training=False)
    acc(pred_labels, labels)

print('VALIDATION ACCURACY : %.2f%%' % (acc.result().numpy() * 100.))

if acc.result().numpy() < .9:
    
    acc.reset_states()
    
    for epoch in range(10):

        step = 0.
        rolling_loss = 0.0

        for images, labels in imagenette:
            with tf.GradientTape() as tape:
                pred_labels, _, _ = model(images)
                loss = cce(pred_labels, labels)
                acc(pred_labels, labels)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            step += 1
            
            rolling_loss += loss

            if step % 50 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                print('Seen so far: %s samples' % ((step + 1) * 32))
                print('Accuracy : %.3f' % acc.result().numpy())

        print('\nEPOCH %d | Loss : %.3f' % (epoch + 1, rolling_loss))
        print('Accuracy : %.3f\n' % acc.result().numpy())
        
        model.save_weights('weights.h5')

# Get weights from Dense layer
weights = model.layers[-1].get_weights()[0]

imagenette_validation = tfds.load('imagenette/320px', split=tfds.Split.VALIDATION)

@tf.function
def apply_preproc(example):
    image, label = example['image'], example['label']
    image = tf.image.resize(image, [320, 320])
    label = tf.one_hot(label, depth=10)
    return image, label

imagenette_validation = imagenette_validation.shuffle(buffer_size=1500).map(apply_preproc).batch(16)

classes = ["tench", "English springer", "cassette player",
           "chain saw", "church", "French horn", "garbage truck", "gas pump",
           "golf ball", "parachute"]

# Class activation maps
for images, labels in imagenette_validation.take(1):
    pred_labels, avg_pooled, feature_maps = model(images, training=False)
    fig, axes = plt.subplots(16, 10, sharex=True, sharey=True, figsize=(25, 30))
    for img_ind in range(16):
        best_class = np.argmax(pred_labels[img_ind])
        BEST_MAP = feature_maps[img_ind] @ weights[:, best_class].reshape(feature_maps.shape[-1], 1)
        for class_ind in range(10):
            axes[img_ind, class_ind].imshow(images[img_ind].numpy().astype(int))
            class_weight = weights[:, class_ind].reshape(feature_maps.shape[-1], 1)
            
            CAM = (feature_maps[img_ind] @ class_weight)
            upsampled_cam = tf.image.resize(CAM, [320, 320]).numpy().reshape(320, 320)
            
            upsampled_cam /= np.max(BEST_MAP)
            upsampled_cam[upsampled_cam < .5] = 0.
            print(np.max(upsampled_cam), end=' ')
            
            axes[img_ind, class_ind].imshow(upsampled_cam, alpha=0.3, cmap="magma_r", vmin=0, vmax=1, interpolation=None)
            
            axes[img_ind, class_ind].set_title("%s - %.1f%%" % (classes[class_ind], 100. * pred_labels[img_ind, class_ind].numpy()))
            
            axes[img_ind, class_ind].set_xticks([])
            axes[img_ind, class_ind].set_yticks([])
        
        print('\n\n')
    fig.tight_layout()
    plt.savefig('heatmaps.png', dpi=300)
        
