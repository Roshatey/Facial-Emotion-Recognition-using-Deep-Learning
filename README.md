# Facial Emotion Recognition using Deep Learning

This project aims to classify the emotion on a person's face into one of seven categories using deep convolutional neural networks (CNNs). 
The model is trained on the FER-2013 dataset, published at the International Conference on Machine Learning (ICML).
The dataset contains 35,887 grayscale images of size 48x48 pixels, labeled with seven emotions:
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised

Using a simple 4-layer CNN, the model achieves 63.2% test accuracy after 50 epochs.

## Dependencies
  - Python 3, OpenCV, Tensorflow
  - To install the required packages, run pip install -r requirements.txt.

## Data Preparation (optional)
The original FER2013 dataset in Kaggle is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing.
In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the dataset_prepare.py file which can be used for reference.

## Algorithm
  1. First, the haar cascade method is used to detect faces in each frame of the webcam feed.
  2. The region of image containing the face is resized to 48x48 and is passed as input to the CNN.
  3. The network outputs a list of softmax scores for the seven classes of emotions.
  4. The emotion with maximum score is displayed on the screen.

## Basic Usage
The repository is currently compatible with tensorflow-2.0 and makes use of the Keras API using the tensorflow.keras library.
First, clone the repository and enter the folder
```bash
git clone https://github.com/Roshatey/Facial-Emotion-Recognition-using-Deep-Learning.git
cd Facial-Emotion-Recognition-using-Deep-Learning
```
Download the FER-2013 dataset inside the src folder.

If you want to train this model, use:
```bash
cd src
python emotions.py --mode train
```
If you want to view the predictions without training again, you can download the pre-trained model from here and then run:
```bash
cd src
python emotions.py --mode display
```
The folder structure is of the form:
src:
  - data (folder)
  - emotions.py (file)
  - haarcascade_frontalface_default.xml (file)
  - model.h5 (file)
This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

To install the required packages:
```bash
pip install -r requirements.txt
```
<img width="1500" height="500" alt="accuracy" src="https://github.com/user-attachments/assets/014e11b0-819b-4b8e-88fd-d3c14b29c5ca" />

