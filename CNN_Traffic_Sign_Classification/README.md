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

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/labels.PNG)

### Data Preprocessing
#### Data Distribution
The distribution of train, validation, and test sets are visualized below. The distribution for train set is not uniform as there is a large difference in number of images available for some of the classes. For instance, there are 2010 images for class Id 2, but only 180 images for class Id 0. Although there is a big disparity between the number of images for each class in the train set, the distribution is also mimicked by data in validation set and test set. Therefore, data
augmentation might help train a less biased model, but its benefit may not reflect on test or validation set accuracies.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Histogram%20of%20image%20data%20distribution.PNG)

### Bounding Boxes
Random images from the dataset along with bounding boxes are plotted in the image below. The bounding boxes however only sometimes correctly demarcates traffic sign in the image properly. In some of the images below the bounding boxes also contain large portions of the image that are outside the traffic signs, and sometimes cuts out the edges of a sign.
<!--
![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Train%20Set%20with%20bounding%20boxes.PNG =100x100)
-->

<img src="https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Train%20Set%20with%20bounding%20boxes.PNG" width="400" height="400" />

Since the images are already cropped with the traffic signs at their center, there is no need to crop the images again the with bounding boxes given. For rest of this project these bounding boxes were ignored because the images are cropped well, and the bounding boxes provided are inaccurate.

### Data Augmentation
Data augmentation was performed to equalize distribution of images in the train set to avoid model bias towards some class of data. A few augmentations were tested using some custom functions and, using the ImageDataGenerator() function from the keras library.</br>
The image below compares the original image on the left, with the same image augmented for brightness and rotation using custom functions (in the middle) and using keras on the right.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Comparison%20of%20augmentation%20using%20custom%20functions%20and%20keras.PNG)

The following images compare augmented data with nearest-fill (left) and zero-fill (right) options. The sharp edges of the zero-fill option could affect our CNN models negatively. This was also found to be the case when both the augments were tested as train set for LeNet CNN.

<img src="https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Augmentation%20with%20nearest%20fill%20(left)%20and%20zero-fill%20(right).PNG" width="850" height="400" />

Following combinations of train set data was tested on a LeNet to see how much effect each of
them could have on model performance. Each model was run three times to ensure that no
output was a result of a bad minima encountered by chance.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Augments%20tested.PNG)

