# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras_preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# from read_data import train_image, train_label, test_image, test_label

PATH = "model_djt.h5"
WIDTH = 256
HIGHT = 256
TOTAL_TRAIN = 3150
TOTAL_TEST = 350
cwd = os.getcwd()
# 改一下路径
root = cwd + "./"

train_filename = []


def input():
    for i in range(1, 11):
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
            lab_one_hot = np.zeros((5,), dtype=np.int)
            lab_one_hot[lab_] = 1
            test_image.append(img_)
            test_label.append(lab_one_hot)
        test_image = np.array(test_image)
        test_label = np.array(test_label)
        coord.request_stop()
        coord.join(threads)
    return train_image, train_label, test_image, test_label


def VGG16(input_tensor=None, input_shape=None, classes=5):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, (12, 12), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape)(
        img_input)
    x = Conv2D(32, (12, 12), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_bn1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_bn1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_bn1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = BatchNormalization(name='block6_bn1')(x)
    x = GlobalMaxPooling2D()(x)

    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name='vgg16')
    return model


if __name__ == '__main__':
    train_image, train_label, test_image, test_label = input()

    model = VGG16(input_tensor=None, input_shape=(256, 256, 1), classes=5)

    model.summary()

    from keras.optimizers import Adam

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

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
                   TensorBoard("./input/log")],
        validation_steps=100)
