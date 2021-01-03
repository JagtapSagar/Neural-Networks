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

