# CMPS 6730 Sample Project

The goal of this project was to evaluate the performances of different machine learning models on a text classification task. The task was to predict whether a comment in the subreddit r/MkeBucks's postgame thread followed a win or a loss.

Four different models were build and evaluated on this task:

- Bernoulli Naive Bayes
- Logistic Regression
- Convolutional Neural Network
- MiniBERT

These models were evaluated according to their F1 score.

### Bernoulli Naive Bayes

The Bernoulli Naive Bayes model was built using scikit-learn's 

### Demo Screenshots

Below are screenshots of a Demo designed to predict whether a comment followed a win or a loss according to the selected model. Here are the predictions for the comment "This team is extremely talented, but I am worried about the three point shooting, and I am skeptical going forward."

Here are the predictions for each model on this comment:

![Naive Bayes](docs/NBDemo.png)
![Logistic Regression](docs/LRDemo.png)
![CNN](docs/CNNDemo.png)
![BERT](docs/BERTDemo.png)
