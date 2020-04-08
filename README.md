# vggface_transfer_learning
This repository contains source code for transfer learning on vgg-face network on custom dataset.

The objective of this mini-project is to test how well transfer learning will perform for facial recognition task. The model being tested is VGG16 Face, that had been trained on millions of faces of different people. 

A small dataset was collected consisting of 11 people (classes) to train and test the model.

The weights used to do transfer learning were converted from MatConvNet to Python Keras, Tensorflow background, and can be found in [link to weights](https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5).

The file vgg-test.py contains the source for testing the network in predicting a particular picture (ex: "test_image.jpg"). The weights files after training are attached on google drive and can be found on https://drive.google.com/open?id=1A_6-9njvQuyXvMW-6N-a7Uawk1ivUuTu.
