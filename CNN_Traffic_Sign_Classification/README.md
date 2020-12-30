Traffic sign classification using CNN
===
### Objective:

The main goal of this project is to perform Traffic Sign Classification from cropped images of traffic signs such as speed limits, stop signs, and right turn ahead to name a few. This sort of classification is performed very commonly in the autonomous vehicle industry. This project aims to implement various Convolution Neural Network based deep learning algorithms, benchmark their performance, and produce a high performing model.

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

The results for each of the above testes are presented below.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Comparison%20of%20performance%20of%20different%20train%20data.PNG)

Inferring from the results above, normalized train set with non-uniform distribution performs
equally well as the distribution equalized with augment 1 configuration. This is could be
because the same initial distribution is consistent with the distribution of validation and test
sets. For this project, the data prepared with augment 1 will be used for further training.


### Train, Validation and Test Data
After applying Augment 1 (Table 1), the augmented train set was combined with the train initial images, shuffled, normalized, and stored as a pickle file. The augments were performed using keras ImageDataGenerator function and is illustrated in the code snippet below.
<!--
![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Generate%20augmented%20image%20set%20for%20histogram%20equalization.PNG)
-->

```
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=13,brightness_range=[0.2,1.8])

# Create augmented images
imageGen = []
labelGen = []
for num in range(0,43,1):
    if np.size(np.where(train_labels==num),axis=1) < maxSize:
        print('[INFO] Augementing images for label: ' + '[' + str(num) + '] ' + labels['SignName'][num])
        indices = np.where(train_labels==num)
        # train_labels[train_labels.tolist().index(num):train_labels.tolist().index(num)+np.size(indices,axis=1)]

        # Prepare data to sample
        data = train_images[train_labels.tolist().index(num):train_labels.tolist().index(num)+np.size(indices,axis=1)]
        samples = data
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate batch of images

        count = 0
        # Generate images to equilize histogram
        for epoch in range(maxSize - np.size(np.where(train_labels==num),axis=1)):
            count += 1
            batch = it.next()
            imageGen.append(np.squeeze(batch.astype('uint8')))
            labelGen.append(num)

        print('[INFO] Generated ' + str(count) + ' images')
```

Each time the test and validation sets are loaded, they are normalized using the same normalization method used for train set. The following code shows the implementation of this.

<!--
![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Load%20Train%2C%20Validation%2C%20and%20Test%20sets.PNG)
-->

```
# Load augmented data images
data = pickle.load(open(DATA_DIR + 'data_aug_1.pickle', 'rb'))

# Extract Train Images
train_images = data['images']
train_labels = data['labels']

# Load validation and test files
valid = pickle.load(open(DATA_DIR + 'valid.pickle', 'rb'))
test = pickle.load(open(DATA_DIR + 'test.pickle', 'rb'))
labels = pd.read_csv(DATA_DIR + 'label_names.csv')

# Extract validation and test image and label sets
valid_images = valid['features']
valid_labels = valid['labels']
test_images = test['features']
test_labels = test['labels']

# Normalize validation and test set
valid_images = preprocess_data(valid_images,norm_255=True)
test_images = preprocess_data(test_images,norm_255=True)
```

The distribution of the data used for train, validation, and test is illustrated below.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Distribution%20of%20X%20(train)%2C%20validation%2C%20and%20test%20set.PNG)

### Benchmarking CNN’s
A variety of Convolutional Neural Network implementations were tested. The networks ranged from shallow CNN’s such as LeNet to modern deep learning networks such as ResNets, and DenseNets. Initially each network was trained three times to ensure that it can output consistent performance. However, when it came to training deeper networks due to computational complexity they were only trained once. </br>
The deeper networks like ResNets and DenseNets were trained using weights from ImageNet as a start point and then the networks were fine-tunned.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Benchmarking%20results.%20(TF%20%3D%20Transfer%20Learning%2C%20CR%20%3D%20Cyclic%20cosine%20learning%20rate).PNG)

### Ensemble
An ensemble of 5 models was trained for LeNet and MiniVGG Net models. Three ensemble tests were performed for each model, all with different train set configuration.
*	Ensemble 1: Trained with Augment 1 generated data.
*	Ensemble 2: Trained with Original (Pre-augmentation) data, unnormalized, augmenting it in the model.
*	Ensemble 3: Trained with Original (Pre-augmentation) data, 255 normalized, augmenting it in the model.
The results of model averaging are presented in the table below.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Ensemble%20(Model%20Averaging%3B%20Train%2C%20Validation%20and%20Test%20accuracy).PNG)

The ensemble test performed servers two purposes. The first it confirms MiniVGG Net is the model to choose for this project because it returns the highest accuracy. And second it also confirms that augment 1 is a good choice to proceed with since it returned higher validation accuracy on most of the models used for ensemble.</br>
The ensembles however do not seem very promising in improving the base performance of the model as both LeNet and MiniVGGNet models trained without the ensembles returned equivalent results.</br>
Comparing classification report between both ensembled and individual model did not seem to improve overall network accuracies by any significant amount either. Even if it did so, this ensemble is very expensive to train several final models as it will likely take a very long time to train each model before averaging them.

### Model Selection
Based on the results presented above, MiniVGG was chosen as the CNN architecture to train models for this project. The reason being that the network has shown to achieve highest test and validation accuracies, and the model may be potentially overfitting the data with consistent ~99.90% train set accuracy. Therefore, there might be scope for improvement with hyperparameter tunning. </br>
Results of tests on Mini-VGGNet and VGG16Net were almost identical. But Mini-VGG Net consistently gave slightly higher test set and validation results. Following table shows the final network architecture of our Mini-VGG Net after hyperparameter tunning (in the next section).

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/CNN_Traffic_Sign_Classification/Images/Mini-VGG%20Network%20Architecture%20used%20for%20the%20final%20model.PNG)

### Hyper parameter Tuning
Once the Mini-VGG Net was selected for the project, hyperparameter tuning was performed for the following hyperparameters.

* Optimizers
* Weight Initialization
* Kernel Sizes
* Convolution Layer Sizes
* Loss Function
* Dense Layer Size
* Pool and stride size
* Dropout for dense layer
* Dropout for pooling layers
* Kernel Regularization
* Activity Regularization
During tuning, ensemble method was used to get multiple results for most hyperparameter inputs.

#### Optimizers
The Results on optimizer tests with default parameters are presented in the table below.

<image>

Adam and RMSprop optimizers showed the best results with default parameters. Based on the above results Adam optimizer was selected for further tuning.

#### Weight Initialization
Following weight initializations were tested for all the layers in the network.
*	Random Normal
*	Random Uniform
*	Truncated Normal
*	Zeros
*	Ones
*	Glorot Normal
*	Glorot Uniform
*	Orthogonal
*	Variance Scaling
Accuracy scores of train, validation, and test sets obtained for respective initialization methods can be seen in the table below. Based on the results of this test Random Uniform weight initialization method chosen.

<image>

#### Activation Function
A list activation functions consisting of ReLU, Softmax, Tanh, and Sigmoid activation functions were tested on the output layer. The following Figure illustrates the train and validation set accuracy for each of these activations applied to the output layer.

<image>

Softmax activation was chosen because it returned the highest accuracies and it also seems to begin fitting to train set much earlier.</br>
Although, ReLU should be the best fit for all initial layers, different activation functions were also tested on all layers. The following figure illustrates the results of these tests.

<image>

Similar results can be seen from the second test; however it must be noted that the networks that were able to fit in this second test did so slower than when ReLU was applied to all initial layers and Softmax to the final.</br>
Therefore, picking ReLU activation function for all initial layers and Softmax activation function for the output layer.

#### Kernel Sizes
Combination of kernel sizes ranging from (3,3) to (11,11) was tested on all four convolution layers in the Mini-VGG Net model. The results of kernel tuning are presented in the table below.

<image>

Based on the results above following kernel size combination was selected.
*	Convolution layer 1: (3,3)
*	Convolution layer 2: (5,5)
*	Convolution layer 3: (3,3)
*	Convolution layer 4: (3,3)

#### Convolution Layer Size
Combination of convolution layer sizes from the list:  [32, 64, 128, 256, 512]  was tested on all four convolution layers in the Mini-VGG Net model. The results of convolution layer size tuning are presented in the table below.

<image>

Layer size combination of (all 128), (32, 64, 128, 256), and (64, 128, 256, 512) are almost identical in performance. Selecting (64, 128, 256, 512) because it has the highest validation set accuracy, even though it is by a very small margin.

#### Dense Layer Size
Since Mini-VGG Net has one dense layer before the output layer, sizes of only that layer can vary as the output layer size must be fixed to the number of labels. Results of dense layer test are presented in the table below.

<image>

Based on the results of this tuning, a dense layer size of 128 was chosen. It provided one of the highest accuracies with only 4194432 number of parameters to train compared to 7373025 parameters for dense layer of size 225, and 5734575 parameters for dense layer of size 175.

#### Pool and Stride
A variety of combination of pool size and stride size was tested. The following tables shows training results for P1, S1 and P2,S2 (max Pooling and stride length) sizes for both the pooling layers.

<image>

Pool size (3,3) and stride length (1,1) were selected for both pool layers because it gave the one of the highest test and validation accuracies.

#### Loss Function
Three loss function were tested. Cross-entropy based loss functions performed the best and the training results are presented in the table below. Categorical cross-entropy, which is also the default loss function used by keras for multiclass classification, was taken.

<image>

#### Dense Layer Dropout
Training for dropout values while changing the dropout rate was performed for dense layer and pooling layers. This section deals with dropout layers for the dense layer only. The table below presents the training results for a range of dropout values.

<image>

All values except for 0.4 seem to be are above 98% test set accuracy. However, this seems like a one-off instance since training with 0.4 dropout rate multiple time reveals that it can consistently output the highest performance. This result is presented in the table below.

<image>

#### Pool Layer Dropouts
This section deals with dropout layers for the pooling layers. The two tables below present the training results for a range of dropout values for the first (top) and the second (bottom) pooling layers, respectively. 

<image>
<image>

Changing dropout values for pooling layers did not seem to affect model performance. And the model seems to perform the worst for very high dropout values. Taking both dropout values 0.25.

#### Kernel and Activity Regularization
Mini-VGG Net employees batch regularization through all the layers in the network, which on its own has a regularizing effect. Therefore, adding additional Ridge, LASSO, or Elastic regression parameters may not be useful. However, just to be sure a range of values for both kernel regularization and activity regularization were tested using L1 and L2 parameters.</br>
The table below shows the results for L1 kernel regularization.

<image>

The table below shows the results for L2 kernel regularization.
 
<image>

The table below shows the results for L1 activity regularization.

<image>

The table below shows the results for L2 activity regularization.

### Snapshot Ensemble
Snapshot ensemble was tested on the final model. In one version the ensemble was tested with Adam optimizer, which is the default optimizer for our model with its default (0.001) learning rate as the start point. In the second, it was tested with SGD optimizer with default (0.01) learning rate as the initial point.
</br>
The image on the left below shows the results of snapshot ensemble using Adam optimizer, and the one on the right is using the SGD optimizer.

<image>

From the SGD plot above, it looks like the initial learning rate was too small. Therefore another snapshot ensemble test was run with a larger SGD initial learning rate of 0.1 and 1. The image below illustrates its output.
  
<image>

As is evident from the two ensembles above; even with a large initial learning rate at the beginning of each learning rate cycle, neither the cost function nor the accuracies change much unless the learning rate is exceptionally high. This possibly indicates that the model potentially has a very wide or flat local/global minima that it tends to stay in.

### Final Model Training and Results
The final model was trained on Mini-VGG Net (illustrated in Table 3) with the hyperparameters that was found via all the testing done in the section 7 above. 

<image>

The final model was trained with above hyperparameters and the figures below illustrate the Train and Validation Set Accuracy (left), and Loss (right).

<image>

Final Train, Validation, and Test Set accuracies are shown in the table below.

<image>

A classification report for all the three image sets was generated and can be found under Appendix: in section 11.

### Conclusion
The Mini-VGG Net based CNN could learn to generalize traffic very well and the objectives of this project were achieved. For many real-world applications however, real time predictions are required. Such as for use in autonomous vehicles. This means that for such specific applications fasters CNN’s that can perform object detection and classification almost simultaneously are required. </br>
Most real-time CNN’s such as faster R-CNN, YOLO and its variants are more suitable for such applications. But these networks usually fall behind in prediction performance with YOLOv2 boasting only around 70-80% accuracy. CNN’s such as the one trained in this project can be used to provide a feedback to real-time CNN algorithms like YOLO to improve its weights in real-time.

### Appendix
#### Classification report for Train set

```
 Train accuracy
                                                    precision    recall  f1-score   support

                              Speed limit (20km/h)       1.00      1.00      1.00      2010
                              Speed limit (30km/h)       1.00      0.99      1.00      2010
                              Speed limit (50km/h)       1.00      1.00      1.00      2010
                              Speed limit (60km/h)       1.00      0.99      1.00      2010
                              Speed limit (70km/h)       1.00      1.00      1.00      2010
                              Speed limit (80km/h)       0.99      1.00      0.99      2010
                       End of speed limit (80km/h)       1.00      1.00      1.00      2010
                             Speed limit (100km/h)       1.00      1.00      1.00      2010
                             Speed limit (120km/h)       1.00      1.00      1.00      2010
                                        No passing       1.00      1.00      1.00      2010
      No passing for vehicles over 3.5 metric tons       1.00      1.00      1.00      2010
             Right-of-way at the next intersection       1.00      1.00      1.00      2010
                                     Priority road       1.00      1.00      1.00      2010
                                             Yield       1.00      1.00      1.00      2010
                                              Stop       1.00      1.00      1.00      2010
                                       No vehicles       1.00      1.00      1.00      2010
          Vehicles over 3.5 metric tons prohibited       1.00      1.00      1.00      2010
                                          No entry       1.00      1.00      1.00      2010
                                   General caution       1.00      1.00      1.00      2010
                       Dangerous curve to the left       1.00      1.00      1.00      2010
                      Dangerous curve to the right       1.00      1.00      1.00      2010
                                      Double curve       1.00      1.00      1.00      2010
                                        Bumpy road       1.00      1.00      1.00      2010
                                     Slippery road       1.00      1.00      1.00      2010
                         Road narrows on the right       1.00      1.00      1.00      2010
                                         Road work       1.00      1.00      1.00      2010
                                   Traffic signals       1.00      1.00      1.00      2010
                                       Pedestrians       1.00      1.00      1.00      2010
                                 Children crossing       1.00      1.00      1.00      2010
                                 Bicycles crossing       1.00      1.00      1.00      2010
                                Beware of ice/snow       1.00      1.00      1.00      2010
                             Wild animals crossing       1.00      1.00      1.00      2010
               End of all speed and passing limits       1.00      1.00      1.00      2010
                                  Turn right ahead       1.00      1.00      1.00      2010
                                   Turn left ahead       1.00      1.00      1.00      2010
                                        Ahead only       1.00      1.00      1.00      2010
                              Go straight or right       1.00      1.00      1.00      2010
                               Go straight or left       1.00      1.00      1.00      2010
                                        Keep right       1.00      1.00      1.00      2010
                                         Keep left       1.00      1.00      1.00      2010
                              Roundabout mandatory       1.00      1.00      1.00      2010
                                 End of no passing       1.00      1.00      1.00      2010
End of no passing by vehicles over 3.5 metric tons       1.00      1.00      1.00      2010

                                          accuracy                           1.00     86430
                                         macro avg       1.00      1.00      1.00     86430
                                      weighted avg       1.00      1.00      1.00     86430

```

#### Classification report for Validation set
```
 Validation accuracy
                                                    precision    recall  f1-score   support

                              Speed limit (20km/h)       1.00      1.00      1.00        30
                              Speed limit (30km/h)       1.00      0.99      1.00       240
                              Speed limit (50km/h)       1.00      1.00      1.00       240
                              Speed limit (60km/h)       1.00      0.96      0.98       150
                              Speed limit (70km/h)       0.99      1.00      0.99       210
                              Speed limit (80km/h)       0.97      0.99      0.98       210
                       End of speed limit (80km/h)       1.00      1.00      1.00        60
                             Speed limit (100km/h)       0.92      1.00      0.96       150
                             Speed limit (120km/h)       0.99      0.99      0.99       150
                                        No passing       0.99      1.00      1.00       150
      No passing for vehicles over 3.5 metric tons       1.00      1.00      1.00       210
             Right-of-way at the next intersection       1.00      1.00      1.00       150
                                     Priority road       1.00      1.00      1.00       210
                                             Yield       1.00      1.00      1.00       240
                                              Stop       1.00      1.00      1.00        90
                                       No vehicles       1.00      1.00      1.00        90
          Vehicles over 3.5 metric tons prohibited       1.00      1.00      1.00        60
                                          No entry       1.00      1.00      1.00       120
                                   General caution       0.98      0.99      0.99       120
                       Dangerous curve to the left       1.00      1.00      1.00        30
                      Dangerous curve to the right       1.00      0.93      0.97        60
                                      Double curve       1.00      0.87      0.93        60
                                        Bumpy road       1.00      1.00      1.00        60
                                     Slippery road       0.91      0.98      0.94        60
                         Road narrows on the right       1.00      0.97      0.98        30
                                         Road work       0.99      1.00      1.00       150
                                   Traffic signals       1.00      1.00      1.00        60
                                       Pedestrians       1.00      1.00      1.00        30
                                 Children crossing       1.00      1.00      1.00        60
                                 Bicycles crossing       1.00      1.00      1.00        30
                                Beware of ice/snow       0.97      1.00      0.98        60
                             Wild animals crossing       0.98      1.00      0.99        90
               End of all speed and passing limits       1.00      1.00      1.00        30
                                  Turn right ahead       1.00      1.00      1.00        90
                                   Turn left ahead       1.00      1.00      1.00        60
                                        Ahead only       1.00      1.00      1.00       120
                              Go straight or right       1.00      1.00      1.00        60
                               Go straight or left       1.00      1.00      1.00        30
                                        Keep right       1.00      1.00      1.00       210
                                         Keep left       1.00      1.00      1.00        30
                              Roundabout mandatory       0.98      0.78      0.87        60
                                 End of no passing       1.00      0.97      0.98        30
End of no passing by vehicles over 3.5 metric tons       0.97      1.00      0.98        30

                                          accuracy                           0.99      4410
                                         macro avg       0.99      0.99      0.99      4410
                                      weighted avg       0.99      0.99      0.99      4410

```

#### Classification report for Test set
```
 Test accuracy
                                                    precision    recall  f1-score   support

                              Speed limit (20km/h)       1.00      1.00      1.00        60
                              Speed limit (30km/h)       1.00      0.99      1.00       720
                              Speed limit (50km/h)       1.00      1.00      1.00       750
                              Speed limit (60km/h)       1.00      0.94      0.97       450
                              Speed limit (70km/h)       0.99      0.99      0.99       660
                              Speed limit (80km/h)       0.95      1.00      0.97       630
                       End of speed limit (80km/h)       0.99      0.95      0.97       150
                             Speed limit (100km/h)       1.00      1.00      1.00       450
                             Speed limit (120km/h)       0.98      1.00      0.99       450
                                        No passing       1.00      1.00      1.00       480
      No passing for vehicles over 3.5 metric tons       1.00      1.00      1.00       660
             Right-of-way at the next intersection       1.00      1.00      1.00       420
                                     Priority road       1.00      0.99      0.99       690
                                             Yield       1.00      1.00      1.00       720
                                              Stop       1.00      1.00      1.00       270
                                       No vehicles       0.97      1.00      0.98       210
          Vehicles over 3.5 metric tons prohibited       1.00      1.00      1.00       150
                                          No entry       1.00      1.00      1.00       360
                                   General caution       0.99      0.97      0.98       390
                       Dangerous curve to the left       0.98      1.00      0.99        60
                      Dangerous curve to the right       0.91      1.00      0.95        90
                                      Double curve       1.00      0.99      0.99        90
                                        Bumpy road       0.94      0.98      0.96       120
                                     Slippery road       0.97      1.00      0.99       150
                         Road narrows on the right       0.98      0.99      0.98        90
                                         Road work       1.00      0.99      0.99       480
                                   Traffic signals       0.96      0.96      0.96       180
                                       Pedestrians       0.95      1.00      0.98        60
                                 Children crossing       0.99      1.00      0.99       150
                                 Bicycles crossing       0.98      1.00      0.99        90
                                Beware of ice/snow       1.00      0.91      0.95       150
                             Wild animals crossing       1.00      0.99      0.99       270
               End of all speed and passing limits       0.97      1.00      0.98        60
                                  Turn right ahead       1.00      1.00      1.00       210
                                   Turn left ahead       1.00      0.99      1.00       120
                                        Ahead only       1.00      1.00      1.00       390
                              Go straight or right       0.99      1.00      1.00       120
                               Go straight or left       1.00      1.00      1.00        60
                                        Keep right       1.00      0.99      1.00       690
                                         Keep left       1.00      1.00      1.00        90
                              Roundabout mandatory       0.95      1.00      0.97        90
                                 End of no passing       1.00      1.00      1.00        60
End of no passing by vehicles over 3.5 metric tons       1.00      1.00      1.00        90

                                          accuracy                           0.99     12630
                                         macro avg       0.99      0.99      0.99     12630
                                      weighted avg       0.99      0.99      0.99     12630

```
