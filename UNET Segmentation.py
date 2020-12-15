"""# Task 2: Applying Unet to segment the images"""

import tensorflow as tf
# tf.enable_eager_execution()
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# from hilbert import hilbertCurve
import imgaug.augmenters as iaa
import numpy as np

# import albumentations as A
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten, LSTM, \
    BatchNormalization, ReLU, Reshape
from tensorflow.keras.models import Model
import random as rn

!pip
install
imgaug

from tensorflow.keras.layers import Flatten

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data_df, test_size=0.12, random_state=42)
X_train.shape, X_test.shape

!pip
install - U
segmentation - models == 0.2
.1

# we are importing the pretrained unet from the segmentation models
# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from segmentation_models import Unet

# sm.set_framework('tf.keras')
tf.keras.backend.set_image_data_format('channels_last')

# loading the unet model and using the resnet 34 and initilized weights with imagenet weights
# "classes" :different types of classes in the dataset
tf.keras.backend.clear_session()
model = Unet('resnet34', encoder_weights='imagenet', classes=21, activation='softmax', input_shape=(256, 256, 3))

model.summary()

# import imgaug.augmenters as iaa
# For the assignment choose any 4 augumentation techniques
# check the imgaug documentations for more augmentations
aug2 = iaa.Fliplr(1)
aug3 = iaa.Flipud(1)
aug4 = iaa.Emboss(alpha=(1), strength=1)
aug5 = iaa.DirectedEdgeDetect(alpha=(0.8), direction=(1.0))


# aug6 = iaa.Sharpen(alpha=(1.0), lightness=(1.5))

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if i == 1:
            plt.imshow(image, cmap='gray', vmax=1, vmin=0)
        else:
            plt.imshow(image)
    plt.show()


# def normalize_image(mask):
#     mask = mask/255
#     return mask

class Dataset:
    # we will be modifying this CLASSES according to your data/problems
    CLASSES = list_classes

    # the parameters needs to changed based on your requirements
    # here we are collecting the file_names because in our dataset, both our images and maks will have same file name
    # ex: fil_name.jpg   file_name.mask.jpg
    def __init__(self, file_names, classes, train_sett):

        ###self.ids = file_names
        # the paths of images
        self.images_fps = list(
            file_names['image'])  ###[os.path.join(images_dir, image_id+'.jpg') for image_id in self.ids]
        # the paths of segmentation images
        self.masks_fps = list(
            file_names['mask'])  ###[os.path.join(images_dir, image_id+".mask.jpg") for image_id in self.ids]
        # the paths of json files
        # self.json_fps     = file_names['json']
        # giving labels for each class
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        self.train_sett = train_sett
        self.w = 256
        self.h = 256

    def __getitem__(self, i):

        # read data

        # self.w, self.h, self.labels_d, self.vertexlist_d = get_poly(self.json_fps[i])

        ###image = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)

        image = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED) / 10
        image_mask = cv2.resize(image_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        ###image_mask = normalize_image(mask)

        image_masks = [(image_mask == v) for v in self.class_values]
        image_mask = np.stack(image_masks, axis=-1).astype('float')

        if self.train_sett == 'train':
            a = np.random.uniform()
            if a < 0.3:
                image = aug2.augment_image(image)
                image_mask = aug2.augment_image(image_mask)
            elif a < 0.6:
                image = aug3.augment_image(image)
                image_mask = aug3.augment_image(image_mask)
            elif a < 0.8:
                image = aug4.augment_image(image)
                image_mask = aug4.augment_image(image_mask)
            else:
                image = aug5.augment_image(image)
                image_mask = image_mask

        return image, image_mask

    def __len__(self):
        return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, len_shape=3527, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len_shape)

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import tensorflow
import keras

optim = keras.optimizers.Adam(learning_rate=0.000086)

focal_loss = sm.losses.cce_dice_loss

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss
# or total_loss = sm.losses.categorical_focal_dice_loss

model.compile(optimizer=optim, loss=focal_loss, metrics=[iou_score])

# Remove Previous Logs
import shutil

shutil.rmtree('logs\Model_Unet', ignore_errors=True)

# Function for Custom Callback for Implementing Micro Averaged F1 Score as the Metric
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def __init__(self, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        # self.X_val, self.y_val = validation_data

    # def on_train_begin(self, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Checking Overfitting after achieving desired val_iou_score
            # if logs.get('val_iou_score')>0.50: #or logs.get('iou_score')>0.50:# and logs.get('accuracy')>0.75:
            # if (logs.get('accuracy')-logs.get('val_accuracy'))>0.06:
            #    print("Required Metric Value Reached, hence terminated at epoch {} to prevent overfitting".format(epoch))
            #    self.model.stop_training = True
            # if logs.get('iou_score')>0.50:
            #     print("Required metric reached, hence terminated at epoch {} to prevent overfitting".format(epoch))
            #     self.model.stop_training = True
            if (logs.get('iou_score') - logs.get('val_iou_score')) > 0.30 and logs.get('iou_score') >= 0.50:
                print("Overfitting started, hence terminated at epoch {} to prevent furthur overfitting".format(epoch))
                self.model.stop_training = True
            return


metrics = Metrics(interval=1)

import keras

# Dataset for train images
CLASSES = list_classes
train_dataset = Dataset(X_train, classes=CLASSES, train_sett='train')
test_dataset = Dataset(X_test, classes=CLASSES, train_sett='test')

train_dataloader = Dataloder(train_dataset, batch_size=16, len_shape=3527, shuffle=True)
test_dataloader = Dataloder(test_dataset, batch_size=16, len_shape=481, shuffle=True)

print(train_dataloader[0][0].shape, train_dataloader[0][1].shape)
BATCH_SIZE = 8
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, 21)

NAME = "Model_Unet"
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME), update_freq='epoch'),
    keras.callbacks.ModelCheckpoint('./best_model_unet.h5', save_weights_only=False, save_best_only=True, \
                                    mode='max', monitor='val_iou_score'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.00000001, patience=2),
    metrics
]

!rm - rf / logs

history_unet = model.fit(train_dataloader, steps_per_epoch=len(train_dataloader),
                         epochs=100, validation_data=test_dataloader, callbacks=callbacks)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history_unet.history['iou_score'])
plt.plot(history_unet.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history_unet.history['loss'])
plt.plot(history_unet.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

X_test_ex = X_test.head(10)
for p in range(10):
    # original image
    image = cv2.imread(list(X_test_ex['image'])[p], cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)

    # predicted segmentation map
    pred_mask = model.predict(image[np.newaxis, :, :, :])
    pred_mask = tf.argmax(pred_mask, axis=-1)

    # original segmentation map
    image_mask = cv2.imread(list(X_test_ex['mask'])[p], cv2.IMREAD_UNCHANGED)
    image_mask = cv2.resize(image_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(10, 6))
    plt.subplot(131)
    plt.imshow(image)  # Original Image
    plt.subplot(132)
    plt.imshow(image_mask, cmap='gray')  # Original Mask
    plt.subplot(133)
    plt.imshow(pred_mask[0], cmap='gray')  # Predicted Mask
    plt.show()