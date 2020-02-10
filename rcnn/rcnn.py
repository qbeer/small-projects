import selectivesearch as ss
import tensorflow as tf
import tensorflow_datasets as tfds

# Using VOC dataset
voc = tfds.load("voc", split=tfds.Split.TRAIN)

for voc_example in voc.take(1):
    image, objects = voc_example['image'], voc_example['objects']

print(image)
print(objects)