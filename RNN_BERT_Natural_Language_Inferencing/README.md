Natural Language Inferencing
---

Natural language processing (NLP) has grown increasingly elaborate over the past few years. Machine learning models tackle question answering, text extraction, sentence generation, and many other complex tasks. But, can machines determine the relationships between sentences, or is that still left to humans? If NLP can be applied between sentences, this could have profound implications for fact-checking, identifying fake news, analyzing text, and much more.</br></br>
If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.</br></br>
The main task here is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.</br></br>
Up until the last few years, Recurrent Neural Networks were considered the go to for NLP related problems. But recently, algorithms such as BERT (Bidirectional Encoder Representations from Transformers) seem to give more promising and consistent results.

### Objective
The goal of this project is to train a set of models to perform the Natural Language Inferencing task and to draw a performance comparison of prediction accuracy and consistency between the RNN based architectures and the BERT transformer-based algorithm.
The RNN based architectures will be tested with a combination of unidirectional and bidirectional SimpleRNN, LSTM, and GRU layers.

### Dataset
There are two datasets used in this project. The links have been included for reference:</br>
* Dataset 1: [Contradictory, My Dear Watson](https://www.kaggle.com/c/contradictory-my-dear-watson/data)</br>
* Dataset 2: [Stanford Natural Language Inference Corpus](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus)
 
Dataset 1 contains 12120 unique examples in the train set. Which consists of premise-hypothesis pairs in fifteen different languages, including Arabic, Bulgarian, Chinese, German, Greek, English, Spanish, French, Hindi, Russian, Swahili, Thai, Turkish, Urdu, and Vietnamese.
Here, the premise provides the context with which the hypothesis sentence will be compared with. The Data set also contains information about what language the text is written in along with the class label for each data point.

Dataset 2 contains The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE). The main aim is to serve both as a benchmark for evaluating representational systems for text, especially including those induced by representation learning methods, as well as a resource for developing NLP models of any kind. 

### Data Analysis
#### Dataset 1
The following pie chart shows the distribution languages of the premise-hypothesis pairs in the dataset.
![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/Language%20distribution.png)

The figure below shows the train set data having the columns id, premise, hypothesis, lang_abv, language, and label. The label contains 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) respectively. 

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/train%20data%201.PNG)

The plot below showcases the distribution of labels in the train set. The distribution is almost equal for all three labels and therefore equalization will not be necessary.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/dataset%201%20label%20distribution.png)

Since this dataset contains multiple languages, we can use any of the freely available language translation API such as google translate to convert all sentence pairs to English. While this may not be necessary will allow us to keep the tokenized corpus short and have fewer redundant words.

#### Dataset 2
As mentioned in an earlier section, this dataset contains about 570k human-written English sentences. Therefore no translation will be necessary. The following image shows the distribution of all the labels in the train set.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/dataset%202%20label%20distribution.png)

Rows with unlabeled data are far and few, and the rows with labeled data are almost equal in distribution. Therefore, no histogram equalization will be needed.

The following word cloud shows some of the most frequently used words from the first 10000 sentence pairs in this dataset.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/word%20cloud.png)

### Preprocessing

The following code was used to clean-up special characters, punctuations, links, and to translate text.
```
# Functions to cleanup special characters and translate

def remove_space(text):
    return " ".join(text.split())

def remove_punctuation(text):
    return re.sub("[!@#$+%*:()'-]", ' ', text)

def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    return soup.get_text()

def remove_url(text):
    return re.sub(r"http\S+", "", text)

def translate(text):
    translator = Translator()
    return translator.translate(text, dest='en').text

def clean_text(text):
    text = remove_space(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    return text
```

#### Translation
The premise-hypothesis sentence pairs were translated using the google translate API, but it was found that the API is inconsistent and fails to translate some sentences. This is especially a problem when one part of a sentence pair is translated to English and the other is not. In this case, it might just be easier to keep sentences in their original language than to use a pair that uses sentences in two different languages.

#### Combining datasets
Both the datasets were combined to create a larger corpus. The majority of this corpus will be in English since the multilingual pairs were already a minority part of the original dataset 1. This combined dataset was used on some of the models that were tested. It was not used with all models because the effects of the non-English sentence pair on classification accuracy are unknown. So the combined dataset was only tested on models that returned consistent prediction accuracy during training.

### Tokenization
Tokenization is the process of creating a dictionary of words from the dataset in order to represent word-based sentences in coordinate-based vector representation. These token-based arrays then can be vectorized and a model can be then trained to classify those vectors.

There are a few ways of performing tokenization. In this project, tokenization was performed using the 'Tokenizer' function from the TensorFlow library. The most common usage of tokenization involves tokenizing only train set. This allows us to get an idea of well the neural network can generalize when testing over the validation or test sets.
This method of implementing tokenizer was implemented in the majority of the models trained. However, to see how much effect this can have a second method of tokenization was implemented in a few models which involves tokenizing the entire dataset.


### Results

The table below shows the final training and validation set accuracy of various models tested.

| Models | Train set accuracy | Validation set accuracy|
|---|---|---|
| Simple NN | 33.7 | 33.6 |
| Single LSTM | 33 | 33 |
| Single Bidirectional LSTM | 55 | 37.7|
| Two Bidirectional LSTM | 46.48 | 39.84|
| Single GRU | 47.30 | 38.94 |
| Two Bidirectional GRU | 45.89 | 40.14 |
| Three Bidirectional GRU | 48.13 | 40 |
| Single Convolution | 37.42 | 34.35 |
| Simple RNN | 48.75 | 35.48 |
| Simple Bidirectional RNN | 44 | 37.38 |
| MiniVGGNet | 98.96 | 52.78 |
| BERT | 85 | 75 |
| LSTM+ SimpleRNN | 41 | 35 |
| Triple LSTM and Tokenization(Method 2) | 95 | 47 |
| Double LSTM and Tokenization(Method 2) | 98 | 42 |
| Simple Bert | 85 | 75.15 |
| BERT (More Epochs) | 98.50 | 75.60 |
| BERT  (Dropout) | 99.56 | 76.50 |
| BERT  (Dataset 1) MultiLinguistic | 98.72 | 60.85 |

Although taking a look at the table above gives us an idea of how these models compare, to get a better understanding we must look into the training and validation accuracy plots.

The following plots show the training and validation set loss and accuracy plots for the model with two bidirectional LSTM and tokenization method 1. It can be observed that the prediction performance while staying close to 39% is inconsistent throughout training while the loss drops smoothly and logarithmically.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/2%20bi%20lstm%20accuracy%20plot.png) ![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/2%20bi%20lstm%20loss%20plot.png)

A similar trend was observed with most other LSTM, GRU, and SimpleRNN based models.

Another approach tested was using simple 1D convolution for simple NLP implementation. Simple shallow convolution-based architectures cannot be expected to perform as well as RNN's. But results from the training that are presented below do show that these models are consistent at the very least and therefore promising when used in deeper networks.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/convolution%20accuracy%20plot.png) ![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/convolution%20loss%20plot.png)

Due to the consistency of the prediction of convolution layers a VGGNet based 1D convolutional network was modeled and trained. The following plots show that a deep convolution-based network can certainly be used as a valid method to perform NLI problem. However, we might not benefit much from tuning a deeper convolutional network as much as a deep RNN based network.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/MiniVGG%20accuracy.png) ![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/MiniVGG%20loss.png)

To check whether the inconsistency in test set accuracy in RNN was due to tokenizing only the train set, the second method of tokenization was used and some of the models were run again. In this method, the entire dataset was tokenized.

The plot on the left below shows the performance plots for a network with two bidirectional LSTM, and the plot on the right shows that of a network with three bidirectional LSTM employed.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/double%20lstm%20with%20complete%20tokenization.png) ![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/triple%20lstm%20with%20complete%20tokenization.png)

These plots show that the size of the corpus used for training has large implications on the network's ability to generalize with consistency.

Finally, a BERT based model was trained for comparison since it is currently the *go-to* algorithm for NLI related problems.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/bert%20initial.png)

The simpler BERT based model easily outperforms shallower RNN's and deep convolution networks in final validation accuracy and consistency of prediction during training epochs and is, therefore, the best-suited algorithm for NLI.

The BERT model was then trained on the multilingual dataset 1 and the combined dataset. The plot on the left below shows the training and validation loss and accuracy for the BERT model trained on dataset 1 alone. And the plot on the right was trained on the combined dataset.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/BERT%20dataset%201.png) ![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/BERT%20combined%20dataset.png)

The combined dataset is much larger and maybe why it results in lower prediction accuracy when trained on a small subset. But it is evident that once tuning has been performed and when a large portion of the dataset is used for training it is going to return high prediction performance.

A little bit of tuning and using a larger set from dataset 2 for tuning resulted in the final BERT model giving us the following performance.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/BERT%20(slightly%20tuned).png)

The following code cell shows the training and validation accuracy along with training time on the final BERT model.

```
Epoch 1/10
2500/2500 [==============================] - 2047s 813ms/step - loss: 0.9012 - accuracy: 0.5414 - val_loss: 0.5506 - val_accuracy: 0.7771
Epoch 2/10
2500/2500 [==============================] - 2034s 814ms/step - loss: 0.4953 - accuracy: 0.8055 - val_loss: 0.4633 - val_accuracy: 0.8241
Epoch 3/10
2500/2500 [==============================] - 2034s 813ms/step - loss: 0.3643 - accuracy: 0.8657 - val_loss: 0.4434 - val_accuracy: 0.8336
Epoch 4/10
2500/2500 [==============================] - 2034s 814ms/step - loss: 0.2731 - accuracy: 0.9026 - val_loss: 0.5099 - val_accuracy: 0.8324
Epoch 5/10
2500/2500 [==============================] - 2035s 814ms/step - loss: 0.2134 - accuracy: 0.9264 - val_loss: 0.5394 - val_accuracy: 0.8326
Epoch 6/10
2500/2500 [==============================] - 2036s 814ms/step - loss: 0.1649 - accuracy: 0.9459 - val_loss: 0.6430 - val_accuracy: 0.8324
Epoch 7/10
2500/2500 [==============================] - 2037s 815ms/step - loss: 0.1320 - accuracy: 0.9585 - val_loss: 0.7760 - val_accuracy: 0.8332
Epoch 8/10
2500/2500 [==============================] - 2036s 815ms/step - loss: 0.1068 - accuracy: 0.9680 - val_loss: 0.8243 - val_accuracy: 0.8303
Epoch 9/10
2500/2500 [==============================] - 2037s 815ms/step - loss: 0.0885 - accuracy: 0.9742 - val_loss: 0.9735 - val_accuracy: 0.8301
Epoch 10/10
2500/2500 [==============================] - 2038s 815ms/step - loss: 0.0739 - accuracy: 0.9792 - val_loss: 1.0463 - val_accuracy: 0.8323
```

### Conclusion
A comparison of various RNN based architectures was made alongside a few convolution based networks and a BERT model.

The BERT transformer-based algorithm performs prediction with high accuracy and has shown to be more consistent compared to other competing methods. With a little tuning, BERT outperformed all the models that were trained and further tuning may provide very promising results.
