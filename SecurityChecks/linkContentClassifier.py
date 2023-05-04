import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Returns the content of the link while still being secure as some of the links may be malicious
def get_content(link):
    try:
        response = requests.get(link)
        soupContent = BeautifulSoup(response.content, 'html.parser')
        return ' '.join([text.get_text() for text in soupContent.find_all()])
    except:
        return None


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenizes the text using a pretrained BERT tokenizer
def tokenize_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    return inputs


data = pd.read_csv('linkTrainingData.csv')
links = data['link']
labels = data['safe']

preprocessedLinks = [tokenize_text(get_content(link)) for link in links]


# Trains a logistic regression model using 10-fold cross validation
nSplit = 10

kfold = KFold(n_splits=nSplit, shuffle=True, random_state=1)

accuracyScores = []
f1Scores = []
precisionScores = []
recallScores = []

for train, test in kfold.split(preprocessedLinks):
    trainX, testX = preprocessedLinks[train], preprocessedLinks[test]
    trainY, testY = labels[train], labels[test]

    model = LogisticRegression()
    model.fit(trainX, trainY)

    yPredictions = model.predict(testX)
    curAccuracy = metrics.accuracy_score(testY, yPredictions)
    curF1 = metrics.f1_score(testY, yPredictions)
    curPrecision = metrics.precision_score(testY, yPredictions)
    curRecall = metrics.recall_score(testY, yPredictions)

    accuracyScores.append(curAccuracy)
    f1Scores.append(curF1)
    precisionScores.append(curPrecision)
    recallScores.append(curRecall)




# Uses metrics module to print the average accuracy of the model
print('Average accuracy: ', sum(accuracyScores) / len(accuracyScores))
print('Average F1 score: ', sum(f1Scores) / len(f1Scores))
print('Average precision: ', sum(precisionScores) / len(precisionScores))
print('Average recall: ', sum(recallScores) / len(recallScores))

