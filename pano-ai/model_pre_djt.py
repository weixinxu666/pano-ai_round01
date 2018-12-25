# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from keras.layers.advanced_activations import PReLU


# from read_data import train_image, train_label, test_image, test_label

PATH = "model_djt.h5"
WIDTH = 256
HIGHT = 256
TOTAL_TRAIN = 3150
TOTAL_TEST = 350
cwd = os.getcwd()
root = cwd + "./"

train_filename = []


def input():
    for i in range(1, 10):
        train_filename.append("./TFcodeX_" + str(i) + ".tfrecord")
    train_fileNameQue = tf.train.string_input_producer(train_filename)
    reader = tf.TFRecordReader()
    train_key, train_value = reader.read(train_fileNameQue)

    test_filename = ["./TFcodeX_" + str(10) + ".tfrecord", ]
    test_fileNameQue = tf.train.string_input_producer(test_filename)
    test_key, test_value = reader.read(test_fileNameQue)

    features = tf.parse_single_example(train_value, features={'data': tf.FixedLenFeature([WIDTH, HIGHT], tf.float32),
                                                              'label': tf.FixedLenFeature([1], tf.int64),
                                                              'id': tf.FixedLenFeature([1], tf.int64), })
    train_img = features["data"]
    train_lab = features["label"]
    train_lab = train_lab - 1
    train_img = tf.reshape(train_img, [WIDTH, HIGHT, 1])

    features = tf.parse_single_example(test_value, features={'data': tf.FixedLenFeature([WIDTH, HIGHT], tf.float32),
                                                             'label': tf.FixedLenFeature([1], tf.int64),
                                                             'id': tf.FixedLenFeature([1], tf.int64), })
    test_img = features["data"]
    test_lab = features["label"]
    test_lab = test_lab - 1
    test_img = tf.reshape(test_img, [WIDTH, HIGHT, 1])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_image = []
        train_label = []
        for i in range(TOTAL_TRAIN):
            img_, lab_ = sess.run([train_img, train_lab])
            img_ = img_.repeat(3, axis=2)
            lab_one_hot = np.zeros((5,), dtype=np.int)
            lab_one_hot[lab_] = 1
            train_image.append(img_)
            train_label.append(lab_one_hot)
        train_image = np.array(train_image)
        train_label = np.array(train_label)

        test_image = []
        test_label = []
        for i in range(TOTAL_TEST):
            img_, lab_ = sess.run([test_img, test_lab])
            img_ = img_.repeat(3, axis=2)
            lab_one_hot = np.zeros((5,), dtype=np.int)
            lab_one_hot[lab_] = 1
            test_image.append(img_)
            test_label.append(lab_one_hot)
            # print(lab_)
        test_image = np.array(test_image)
        test_label = np.array(test_label)

        coord.request_stop()
        coord.join(threads)
    return train_image, train_label, test_image, test_label


train_image, train_label, test_image, test_label = input()
print(train_image.shape, train_label.shape)
print(test_image.shape, test_label.shape)

WEIGHTS_PATH_NO_TOP = "./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
# base_model.load_weights(WEIGHTS_PATH_NO_TOP)

model = Sequential()
model.add(base_model)
model.add(Dense(1024, activation='relu'))
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Dense(256, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation="softmax"))

model.summary()

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_datagen.fit(train_image)

train_generator = train_datagen.flow(
    train_image,
    train_label,
    batch_size=16)

test_datagen = ImageDataGenerator()

test_datagen.fit(test_image)

validation_generator = test_datagen.flow(
    test_image, test_label,
    batch_size=16)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_image) / 16,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint(filepath=PATH, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"),
               EarlyStopping(monitor='val_loss', patience=20, mode='auto'),
               TensorBoard("./")],
    validation_steps=100)