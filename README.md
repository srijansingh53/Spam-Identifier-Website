# Spam-Identifier-Website
A web application deployed on Node.js to classify spam/non-spam messages using Multinomial Naive Bayes algorithm.

## Getting Started
You should have either python 3.x installed on command prompt or Anaconda (recommended). The dataset used is taken from [Kaggle SMS Spam](https://www.kaggle.com/uciml/sms-spam-collection-dataset). 
### Prerequisites

The following libraries are needed to run the application smoothly:

```
scikit-learn
nltk
pickle
numpy
pandas
matplotlib
seaborn
keras (for LSTM training)
```

### Installation

You need to install Node.js in your computer and install some node modules using npm.

The following command intalls the required node modules.
```
npm install express body-parser ejs path 
```

## Running Scripts

A short exploratory data analysis is done to get familiar with the dataset
```
python eda.py
```
![](/images/count.png)
![](/images/histogram_lengthwise.png)

Run this command to train the classifier required for production
```
python MNB_production.py
```
A model with the name MNB.pkl will be saved in the model folder.

## Deployment

The server is created using node with express as its backend. Run app.js to deploy the model on [localhost:3000](http://localhost:3000/)
which run predict.py in the backend.
```
node app.js
```