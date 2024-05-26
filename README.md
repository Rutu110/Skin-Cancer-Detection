# Skin-Cancer-Detection

Overview

This project focuses on the development of a machine learning model to detect skin cancer from images. The model classifies images of skin lesions as either benign or malignant, helping in early detection and treatment of skin cancer.
Features

    Image preprocessing and augmentation
    Convolutional Neural Network (CNN) model for image classification
    Training and evaluation scripts
    User interface for uploading and predicting skin cancer from images

Prerequisites

Before you begin, ensure you have met the following requirements:

    Python 3.7 or higher
    TensorFlow 2.0 or higher
    Other dependencies listed in requirements.txt

Installation

    Clone the repository:

    sh

git clone https://github.com/yourusername/skincancer-detection.git
cd skincancer-detection

Create a virtual environment:

sh

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:

sh

    pip install -r requirements.txt

Dataset

The dataset used for this project can be downloaded from Kaggle. After downloading, place the images in a directory named data/ within the project folder.
Usage
Training the Model

To train the model, use the train.py script:

sh

python train.py --dataset data/ --epochs 50 --batch_size 32

Evaluating the Model

To evaluate the model on a test set, use the evaluate.py script:

sh

python evaluate.py --model model.h5 --dataset data/test/

Predicting with the Model

To predict if a skin lesion is benign or malignant, use the predict.py script:

sh

python predict.py --model model.h5 --image sample_image.jpg

Running the Web Application

A simple web application is provided to upload and predict images using a web interface. To run the web app, use the following command:

sh

python app.py

Then, open your browser and go to http://127.0.0.1:5000/.
