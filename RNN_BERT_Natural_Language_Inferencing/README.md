Natural Language Inferencing
---

Natural language processing (NLP) has grown increasingly elaborate over the past few years. Machine learning models tackle question answering, text extraction, sentence generation, and many other complex tasks. But, can machines determine the relationships between sentences, or is that still left to humans? If NLP can be applied between sentences, this could have profound implications for fact-checking, identifying fake news, analyzing text, and much more.</br></br>
If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.</br></br>
The main task here is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.</br></br>
Up until last few years Recurrent Neural Networks were considered the go to for NLP related problems. But recently, algorithms such as BERT (Bidirectional Encoder Representations from Transformers) seem to give more promissing and consistent results.

### Objective
The goal of this project is to train a set of models to perform the Natural Language Inferencing task and to draw a performance comparison of prediction accuracy and consistency between the RNN based architectures and the BERT transformer based algorithm.
The RNN based architectures will be tested with a combinations of unidirectional and bidirectional SimpleRNN, LSTM and GRU layers.

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

The figure below shows the train set data having the columns id, premise, hypothesis, lang_abv, language and label. The label contains 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) respectively. 

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/train%20data%201.PNG)

The plot below showcases the distribustion of labels in the train set. The distribution is almost equal for all three labels and therefore equalization will not be necessary.

![alt text](https://github.com/JagtapSagar/Neural-Networks/blob/main/RNN_BERT_Natural_Language_Inferencing/Images/dataset%201%20label%20distribution.png)

Since this dataset contains multiple languages, we can use any of the freely available language translation API's such as google translate to convert all sentence pairs to english. While this may not be necessary will allow us to keep the tokenized corpus short and have less redundant words.

#### Dataset 2
As mentioned in earlier section, this dataset contains about 570k human-written English sentences. Therefore no transation of will be necessary. The following image shows distribution of all the labels in the train set.

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
The premise-hpyothesis sentence pairs were translated using the the google translate API, but it was found that the API is inconsistent and fails to translate some sentences. This is especially a problem when one part of a sentence pair is translated to english and the other is not. In this case it might just be easier to keep sentences in their original language than to use a pair that uses sentences in two different languages.

#### Combining datasets
Both the datasets were the combined to create to larger corpus. This majority of this corpus will be in english since the multilingual pairs were already a minority part of the original dataset 1. This combined dataset was used on some of the models that were tested. It was not used with all models because effects of the non-english sentence pair on classification accuracy are unknown. So the combined datast was only tested on models that returned consistent predicyion accuracy during training.

### Tokenization
Tokanization is the process of creating a dictionary of words from the dataset in order to represent word based sentences in coordinate based vector representation. These token based arrays then can be vectorized and a model can be then trained to classify those vectors.

There are a few ways of performing tokenization. In this project tokenization was performed using the 'Tokenizer' function from the tensorflow library. The most uage of takenization involves tokenizing only train set. This allows us get an idea of well the neural network is able to generalize when testing over the validation or test sets.
This method of implementing tokenizer was implemented in majority of the models trained. However, in order to see how much effect this can have a second method of tokenization was implemented in a few models which involves tokenizing the entire dataset.


### Results


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

