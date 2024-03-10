# Bird CNN Classificator

## Table of Contents
* [Technologies](#technologies)
* [Introduction](#introduction)
* [Capuchin bird](#capuchin-bird)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [CNN Model Architecture](#cnn-model-architecture)
* [Model Learning Results](#model-learning-results)
* [Summary](#summary)
* [Acknowledgments](#acknowledgments)

## Technologies
* Python 3.10
* [TensorFlow 2.10](https://www.tensorflow.org/)
* [Keras 2.10](https://keras.io/)

## Introduction
The aim of the project was to build a model based on Convolutional Neural Network (CNN) to classify audio files. The model had the task of detecting and counting the sounds of capuchin birds in an audio recording of several minutes. The project was based on the [work](https://github.com/nicknochnack/DeepAudioClassification) of [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte). The model built by the above-mentioned author fulfils the task set for it, but has one major drawback, a very long learning and classification time. This is due to the lack of preparation of the input data. In this paper, I will try to show how I have dealt with this problem without losing the classification performance of the model.

### Capuchin bird

Capuchin bird image  
![image](https://github.com/Gorwik/Bird-CNN-Classificator/assets/101866409/c087662c-805c-4f81-8cd3-5cbe6ea638fb)

[Capuchin bird sound](https://www.youtube.com/watch?v=tVSvaq0J84M)

## Dataset
More than 800 specially prepared 3-second recordings have been used to teach the model. These recordings are divided into 2 categories. The first are recordings where a single capuchin sound can be heard, while the second contains recordings of other forest sounds, e.g. the singing of other birds, the sound of crickets, etc.  
[Dataset](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)

## Preprocessing
The first step was to reduce the original sampling frequency from 44100 Hz to 22050 Hz. The prepared audio fragments are then processed, where frequencies below 256 HZ and above 2048 HZ are cut out. This is due to the fact that the sound of the capuchin singing is in this frequency range. Mel spectrograms are then created from the recordings to serve as input to our model.  The choice of mel spectrogram as input to the CNN model was dictated by the desire to reduce the number of dimensions and normalise the data while retaining the information contained in the recording.

Capuchine sound spectogram  
![image](https://github.com/Gorwik/Bird-CNN-Classificator/assets/101866409/48639887-1a54-4e4c-974d-5b1cdad3a178)

Capuchine sound mel-spectogram  
![image](https://github.com/Gorwik/Bird-CNN-Classificator/assets/101866409/038305f1-bafd-48f9-9917-0b8c3f798abf)

## CNN Model Architecture
I chose the **AlexNet** architecture as the learning model:  

| **Layer (type)**                           | **Output Shape**    | **Param #** | 
|--------------------------------------------|---------------------|----------------|
| conv2d (Conv2D)                            | (None, 63, 30, 128) | 15616          |
| batch_normalization (BatchNormalization)   | (None, 63, 30, 128) | 512            |
| max_pooling2d (MaxPooling2D)               | (None, 31, 15, 128) | 0              |
| conv2d_1 (Conv2D)                          | (None, 31, 15, 256) | 819456         |
| batch_normalization_1 (BatchNormalization) | (None, 31, 15, 256) | 1024           |
| max_pooling2d_1 (MaxPooling2D)             | (None, 10, 5, 256)  | 0              |
| conv2d_2 (Conv2D)                          | (None, 10, 5,256)   | 590080         |
| batch_normalization_2 (BatchNormalization) | (None, 10, 5, 256)  | 1024           |
| conv2d_4 (Conv2D)                          | (None, 10, 5, 256)  | 65792          |  

| **Total params**     | **5,234,433** |
|----------------------|----------------|
| Trainable params     | 5,232,129      |
| Non-trainable params | 2,304          |


The table above shows that the model I built has 5,234,433 prameters, where the model built by [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte) has 770,482,865. There is a **147-fold** reduction in the number of parameters, which significantly decreases the complexity of the model.

## Model Learning Results
*	Cost_function: 0.091
*	Precision: 0.9866
*	Accuracy: 0.9932
*	Validation Cost_Function: 0.000456
*	Validation Precision: 1
*	Validation Accuracy: 1

Graphs of results
![image](https://github.com/Gorwik/Bird-CNN-Classificator/assets/101866409/892d3cc9-9fe0-4de8-a6ab-86d30546f547)  

## Summary
I prepared the input data as follows:
- reduction of the sampling rate
- application of a band-pass filter
- application of the mel-spectogram

Reducing the sampling rate reduced the input data by half. The application of a band-pass filter helped to eliminate low-frequency noise, on the one hand, and to reduce the proportion of the power spectrum of the songbirds in the higher pitch, on the other. The mel-spectogram, in turn, helped to generalise the features of the input data as well as reduce their size. As a result, I obtained a model with 147 times fewer parameters, which significantly accelerated the learning time of the model as well as its performance of classification while maintaining accuracy.

## Acknowledgments
- Project based on [work](https://github.com/nicknochnack/DeepAudioClassification) and [video](https://www.youtube.com/watch?v=ZLIPkmmDJAc)
- [Audio Preprocessing](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
