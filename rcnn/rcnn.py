import selectivesearch as ss
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

N = 10
MIN_IOU = .5

class RoiPooling(tf.keras.layers.Layer):
    def __init__(self, size=(7, 7)):
        super(RoiPooling, self).__init__()

        self.height = size[0]
        self.width = size[1]

    @tf.function
    def __areas_to_pool(self, region_width, region_height, region_width_step, region_height_step):
        """
            [
                (from_x, from_y, to_x, to_y), ...
            ]
        """
    
        areas = [[(width_ind * region_width_step, height_ind * region_height_step,
                  (width_ind + 1) * region_width_step if (width_ind + 1)  < self.width else region_width,
                  (height_ind + 1) * region_height_step if (height_ind + 1) < self.height else region_height) for width_ind in range(self.width)] for height_ind in range(self.height)]

        return areas

    def __pool_roi(self, feature_map, relative_roi):

        f_width = f_height = 14
        f_x_min = tf.cast(f_width * relative_roi[0], 'int32')
        f_y_min = tf.cast(f_width * relative_roi[1], 'int32')
        f_x_max = tf.cast(f_height * relative_roi[2], 'int32')
        f_y_max = tf.cast(f_height * relative_roi[3], 'int32')

        region = feature_map[f_y_min:f_y_max, f_x_min:f_x_max, :]
        
        region_width = f_x_max - f_x_min
        region_height = f_y_max - f_y_min

        region_width_step = tf.cast(region_width / self.width, 'int32')
        region_height_step = tf.cast(region_height / self.height, 'int32')

        areas = self.__areas_to_pool(region_width, region_height,
                                    region_width_step, region_height_step)
        
        pooled_roi = tf.stack([[
            tf.math.reduce_max(region[block[1]:block[3], block[0]:block[2], :], axis=[0, 1])
            for block in row ]
            for row in areas ])
        
        return pooled_roi

    def __pool_rois(self, feature_map, rois):

        def pool_roi(roi):
            return self.__pool_roi(feature_map, roi)

        return tf.map_fn(pool_roi, rois, dtype=tf.float32) 
    
    def call(self, x):

        def pool_rois(x):
            feature_map, rois = x
            return self.__pool_rois(feature_map, rois)
        
        pooled_rois = tf.map_fn(pool_rois, x, dtype=tf.float32)

        return pooled_rois[0]

    def compute_output_shape(self, input_shape):
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == 1, 'Batch size must be 1'
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[-1]
        return (n_rois, self.height, self.width, n_channels)

# Using VOC dataset
voc = tfds.load("voc", split=tfds.Split.TRAIN)

# Building the model
def build_model():
    input_image = tf.keras.Input(shape=(224, 224, 3))
    rois = tf.keras.Input(shape=(N, 4)) # N for all

    vgg16 = tf.keras.applications.VGG16(weights='imagenet', input_tensor=input_image,
                                    include_top=False, pooling=None)
    vgg16 = tf.keras.models.Sequential(layers=vgg16.layers[:-1])
    roi_pooling = RoiPooling()

    x = vgg16(input_image)

    pooled_rois = roi_pooling((x, rois))
    bbox_preds = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), activation='relu')(pooled_rois)
    bbox_preds = tf.keras.layers.Flatten()(bbox_preds)
    bbox_preds = tf.keras.layers.Dense(256, activation='relu')(bbox_preds)
    bbox_preds = tf.keras.layers.Dense(4)(bbox_preds)
    #print('bbox_preds.shape : ', bbox_preds.shape)

    logits = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), activation='relu')(pooled_rois)
    logits = tf.keras.layers.Flatten()(logits)
    logits = tf.keras.layers.Dense(21)(logits)
    #print('logits.shape : ', logits.shape)

    model = tf.keras.Model(inputs=[input_image, rois], outputs=[logits, bbox_preds])

    return model

pascal_voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def convert_regions_to_relative_rois(regions):
    relative_rois = np.zeros(shape=(N, 4), dtype=np.float32)

    def sortBySize(region):
        return region['size']

    regions.sort(key=sortBySize, reverse=True)

    for ind in range(N):
        x_min, y_min, w, h = regions[ind]['rect']
        x_max = x_min + w
        y_max = y_min + h
        # making it relative
        relative_rois[ind] = np.array([x_min, y_min, x_max, y_max], dtype=np.float32).reshape(4,) / 224.
    
    return relative_rois


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def label_regions(objects, rois):
    true_bboxes = objects['bbox'].numpy()[0]
    labels = objects['label'].numpy()[0]

    roi_labels = []
    rois_labeled = []

    for label, true_bbox in zip(labels, true_bboxes):
        for roi in rois:
            iou = bb_intersection_over_union(true_bbox, roi)
            if iou > MIN_IOU:
                roi_labels.append(tf.one_hot(label, depth=21))
                rois_labeled.append(roi)
            else:
                roi_labels.append(tf.one_hot(0, depth=21))
                rois_labeled.append(roi)
    
    return np.array(rois_labeled)[:N], tf.stack([roi_label for roi_label in roi_labels[:N]])

def fast_rcnn_loss(logits, labels, bbox_preds, bbox):
    classification_loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    smooth_l1 = 0.0

    

    for ind, (bb, lab) in enumerate(zip(bbox.numpy()[0], labels)):
        lab = np.argmax(lab)
        if lab > 0:
            x, y, x_, y_ = bb
            g_w, g_h = x_ - x, y_ - y
            p_x, p_y, p_w, p_h = bbox_preds[ind] + 1e-12
            t_x, t_y, t_w, t_h = (x - p_x) / (p_w + 1e-12), (y - p_y) / (p_h + 1e-12),\
                                tf.math.log(g_w / (p_w + 1e-12)), tf.math.log(g_h / (p_h + 1e-12))
            t = tf.stack([t_x, t_y, t_w, t_h])
            p = tf.stack([p_x, p_y, p_w, p_h])
            L = tf.reduce_sum(tf.abs(t - p))
            smooth_l1 += tf.cond(L < 1, lambda : 0.5 * tf.square(L), lambda : L - 0.5)
        else:
            smooth_l1 += 0.0
    return tf.reduce_mean(classification_loss + smooth_l1)

optimizer = tf.keras.optimizers.Adam()

roi_pooling = RoiPooling()

for example in voc.batch(1).take(1):
    image, objects = example['image'], example['objects']
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)

    _, regions = ss.selective_search(image[0], scale=224, sigma=0.9, min_size=1600)

    vgg16 = tf.keras.applications.VGG16(weights='imagenet',
                                    include_top=False, pooling=None)
    vgg16 = tf.keras.models.Sequential(layers=vgg16.layers[:-1])
    features = vgg16(image)

    rois = convert_regions_to_relative_rois(regions)
    rois, roi_labels = label_regions(objects, rois)

    rois = tf.expand_dims(rois, axis=0)

    #print(image.shape, rois.shape)

    pooled_rois = roi_pooling((image, rois))
    #print(pooled_rois.shape)
    #print(pooled_rois.shape)

model = build_model()

for voc_example in voc.shuffle(buffer_size=1000).batch(1).take(100):
    # relative bboxes are given (!)
    image, objects = voc_example['image'], voc_example['objects']
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    _, regions = ss.selective_search(image[0], scale=224, sigma=0.9, min_size=1000)
    rois = convert_regions_to_relative_rois(regions)
    rois, roi_labels = label_regions(objects, rois)

    rois = tf.expand_dims(rois, axis=0)

    with tf.GradientTape() as tape:
        logits, bbox_preds = model([image, rois])
        #print('logits : ', logits.shape)
        #print('bbox_preds : ', bbox_preds.shape)
        loss = fast_rcnn_loss(logits, roi_labels, bbox_preds, rois)
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    print(loss)
