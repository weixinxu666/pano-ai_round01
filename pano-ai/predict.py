# coding:utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np
from keras.engine.saving import load_model

WIDTH = 256
HIGHT = 256
TOTAL_TEST = 350
cwd = os.getcwd()
root = cwd + "./"

train_filename = []
def input_test(test_path):
    train_filename.append(test_path)
    train_fileNameQue = tf.train.string_input_producer(train_filename)
    reader = tf.TFRecordReader()
    train_key, train_value = reader.read(train_fileNameQue)

    features = tf.parse_single_example(train_value, features={'data': tf.FixedLenFeature([WIDTH, HIGHT], tf.float32),
                                                              'label': tf.FixedLenFeature([1], tf.int64),
                                                              'id': tf.FixedLenFeature([1], tf.int64), })
    train_img = features["data"]
    train_lab = features["label"]
    train_lab = train_lab - 1
    train_img = tf.reshape(train_img, [WIDTH, HIGHT, 1])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_image = []
        train_label = []
        for i in range(TOTAL_TEST):
            img_, lab_ = sess.run([train_img, train_lab])
            img_ = img_.repeat(3, axis=2)
            lab_one_hot = np.zeros((5,), dtype=np.int)
            lab_one_hot[lab_] = 1
            train_image.append(img_)
            train_label.append(lab_one_hot)
        train_image = np.array(train_image)
        train_label = np.array(train_label)
        coord.request_stop()
        coord.join(threads)
    return train_image, train_label

def get_predict_label(path):
    _, labels = input_test(path)
    labels = np.argmax(labels, axis=1)
    return labels + 1

def accuracy_predict(label_predict,labels):
    count = 0
    for i in range(len(label_predict)):
        if label_predict[i] == labels[i]:
            count = count + 1
    acc_predict = count / len(label_predict)
    return acc_predict

def load(path):
    test_image, _ = input_test(path)
    model = load_model("model_djt (1).h5")
    pre = model.predict(test_image,verbose=1)
    pre = np.argmax(pre, axis=1)
    pre = pre + 1
    return pre


def main():
    label = load("TFcodeX_3.tfrecord")
    return label


if __name__ == '__main__':
    label = main()
    print(label)
    print("label:\n", label)
    print(len(label))
    labels_predict = get_predict_label("TFcodeX_3.tfrecord")
    print("predict_label:\n", labels_predict)
    print(len(labels_predict))
    acc_predict = accuracy_predict(labels_predict,label)
    print(acc_predict)
# # coding:utf-8
# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division
#
# import os
# import tensorflow as tf
# import numpy as np
# from keras.engine.saving import load_model
#
# WIDTH = 256
# HIGHT = 256
# TOTAL_TEST = 170
# cwd = os.getcwd()
# root = cwd + "./"
#
# train_filename = []
# def input_test(test_path):
#     train_filename.append(test_path)
#     train_fileNameQue = tf.train.string_input_producer(train_filename)
#     reader = tf.TFRecordReader()
#     train_key, train_value = reader.read(train_fileNameQue)
#
#     features = tf.parse_single_example(train_value, features={'data': tf.FixedLenFeature([WIDTH, HIGHT], tf.float32),
#                                                               'label': tf.FixedLenFeature([1], tf.int64),
#                                                               'id': tf.FixedLenFeature([1], tf.int64), })
#     train_img = features["data"]
#     train_lab = features["label"]
#     train_lab = train_lab - 1
#     train_img = tf.reshape(train_img, [WIDTH, HIGHT, 1])
#
#     with tf.Session() as sess:
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         train_image = []
#         train_label = []
#         for i in range(TOTAL_TEST):
#             img_, lab_ = sess.run([train_img, train_lab])
#             img_ = img_.repeat(3, axis=2)
#             lab_one_hot = np.zeros((5,), dtype=np.int)
#             lab_one_hot[lab_] = 1
#             train_image.append(img_)
#             train_label.append(lab_one_hot)
#         train_image = np.array(train_image)
#         train_label = np.array(train_label)
#         coord.request_stop()
#         coord.join(threads)
#     return train_image, train_label
#
#
# def load(path):
#     test_image, _ = input_test(path)
#     model = load_model("model_djt.h5")
#     pre = model.predict(test_image,verbose=1)
#     pre = np.argmax(pre, axis=1)
#     pre = pre + 1
#     return pre
#
#
# def main():
#     label = load("TFcodeX_test.tfrecord")
#     return label
#
#
# if __name__ == '__main__':
#     label = main()
#     print(label)