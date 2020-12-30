Traffic sign classification using CNN
===
### Objective:

The main goal of this project is to perform Traffic Sign Classification from cropped images of traffic signs such as speed limits, stop signs, and right turn ahead to name a few. This sort of classification is performed very commonly in the autonomous vehicle industry. This project aims to implement various Convolution Neural Network based deep learning algorithms, benchmark their performance, and produce a high performing model.

The project notebook is split into the following hierarchy:

Import and load dataset</br>
Visualization</br>
Preprocessing functions</br>
Data Augmentation</br>
Benchmarking various CNN models</br>
Ensembles</br>
Snapshot Ensemble</br>
Hyperparameter tuning</br>
Final Model

The final model uses MiniVGGNet CNN architecture and gives a Train/Validation/Test set accuracy of 99.90/99.05/99.07 % respectively.


### Dataset:

The dataset contains images was sourced from Kaggle;
https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed
</br>The dataset contains 51839 unique cropped RGB images of traffic signs stored in pickle file format. The images are split into three sets: train set with 34799 images, validation set with 4410 images, and test set with 12360 images. The dataset also contains images stored in different files with augmented images with different normalizations. The dataset also contains bounding box information for all images of the traffic signs.

![alt text](http://url/to/img.png)

