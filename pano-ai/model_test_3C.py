# coding:utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.engine.saving import load_model
import tensorflow as tf
import numpy as np


def test_load_office(path):
    my_dataset = tf.data.TFRecordDataset([path])
    my_dataset = my_dataset.map(analy_TFrecord)
    iterator = my_dataset.make_one_shot_iterator().get_next()
    labels = []
    X_test = []
    with tf.Session() as sess:
        while True:
            try:
                id, label_onehot, pic = sess.run(iterator)
                labels.append(label_onehot)
                pic = np.asarray(pic)
                pic = pic.reshape(256,256,1)
                pic=pic.repeat(3,axis=2)
                labels.append(label_onehot)
                X_test.append(pic)
            except:
                break
    return np.asarray(X_test).reshape(-1, 256, 256, 3), np.asarray(labels)

def analy_TFrecord(input):
    Features = {"id": tf.FixedLenFeature([], tf.int64),
                "data": tf.FixedLenFeature([256, 256], tf.float32),
                "label": tf.FixedLenFeature([], tf.int64)}
    Features = tf.parse_single_example(input, Features)
    return Features["id"], tf.one_hot(Features["label"] - 1, 5), Features["data"]

def model_test(input_data):
    X_test,_ = test_load_office(input_data)
    model = load_model("model_djt.h5")
    prediction = model.predict(X_test,verbose=1)
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction + 1
    return prediction

# coding:utf-8




def get_predict_label(input):
    my_dataset = tf.data.TFRecordDataset([input])
    my_dataset = my_dataset.map(analy_TFrecord)
    iterator = my_dataset.make_one_shot_iterator().get_next()
    labels = []
    with tf.Session() as sess:
        while True:
            try:
                id, label_onehot, pic = sess.run(iterator)
                labels.append(label_onehot)
            except:
                break
        labels = np.argmax(labels, axis=1)
    return labels + 1

def accuracy_predict(label_predict,labels):
    count = 0
    for i in range(len(label_predict)):
        if label_predict[i] == labels[i]:
            count = count + 1
    acc_predict = count / len(label_predict)
    return acc_predict

def main():
    label = model_test("TFcodeX_2.tfrecord")
    return label


if __name__ == '__main__':
    label = main()
    print(label)
    print("label:\n", label)
    print(len(label))
    labels_predict = get_predict_label("TFcodeX_2.tfrecord")
    print("predict_label:\n", labels_predict)
    print(len(labels_predict))
    acc_predict = accuracy_predict(labels_predict,label)
    print(acc_predict)