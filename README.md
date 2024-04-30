# Milwaukee Bucks Text Classification Project

The goal of this project was to evaluate the performances of different machine learning models on a text classification task. The task was to predict whether a comment in the subreddit r/MkeBucks's postgame thread followed a win or a loss.

Four different models were build and evaluated on this task:

- Bernoulli Naive Bayes
- Logistic Regression
- Convolutional Neural Network
- MiniBERT

These models were evaluated according to their F1 score.

### Bernoulli Naive Bayes

The Bernoulli Naive Bayes model was built using Python's \texttt{scikit-learn} library. The comments were tokenized using the \texttt{CountVectorizer()}, adhering to the Bag of Words model framework. For this model, the Reddit comments were stemmed, stopwords were removed, and non alphanumeric characters were removed. A simple modification was made to the default settings of \texttt{CountVectorizer()} and Bernoulli Naive Bayes in \texttt{scikit-learn} by allowing the model to capture bi-grams, as while the goal was to keep the Naive Bayes model simple, accounting for bi-grams made the model more robust and able to capture more nuances within the data.
Below are some evaluation metrics for the model:

- Precision: 0.574
- Recall: 0.953
- F1: 0.717

### Logistic Regression

The Logistic Regression model was built using scikit-learn's LogisticRegression() package, and the model was designed to interpret both unigrams and bigrams.

Below are some evaluation metrics for the model:

- Precision: 0.661
- Recall: 0.749
- F1: 0.703

### Convolutional Neural Network

The Bernoulli Naive Bayes model was built using TensorFlow.

Below are some evaluation metrics for the model:

- Precision: 0.665
- Recall: 0.733
- F1: 0.698

### Conclusions:

### Demo Screenshots

Below are screenshots of a Demo designed to predict whether a comment followed a win or a loss according to the selected model. Here are the predictions for the comment "This team is extremely talented, but I am worried about the three point shooting, and I am skeptical going forward."

Here are the predictions for each model on this comment:

![Naive Bayes](NBDemo.png)

![Logistic Regression](LRDemo.png)

![CNN](CNNDemo.png)

![BERT](BERTDemo.png)

