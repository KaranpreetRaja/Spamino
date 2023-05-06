import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from requests.exceptions import Timeout

# Returns the content of the link while still being secure as some of the links may be malicious
def get_content(link):
    print(link)
    try:
        response = requests.get(link, timeout=2)
        if response.status_code == 200:
            soupContent = BeautifulSoup(response.content, 'html.parser')
            print ("found")
        #print (soupContent)
            str =' '.join([text.get_text() for text in soupContent.find_all(['h1', 'h2', 'h3', 'p'])])
            if str =="":
                return None
            return str
        else:
            print("not ok", response.status_code)
            return None
    
    except Timeout:
        print("Request timed out!")
        return None
    
    except Exception:
        # print(Exception)
        return None

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
max_length = tokenizer.model_max_length



# Tokenizes the text using a pretrained BERT tokenizer
def tokenize_text(text):
    # Truncate the input text to fit within the maximum sequence length
    truncated_text = text[:max_length]
    tokens = tokenizer(truncated_text, truncation=True,
                    max_length=max_length, padding='max_length',
                    return_tensors='pt')
    return tokens

# Read the CSV file
data = pd.read_csv('DataBases/badtest.csv')

print("Done reading")

# Get the content for each URL and store it in a new column
data['content'] = data['URL'].apply(get_content)

print("Done applying links")

# Remove rows where content is None and their corresponding spam values
data = data.dropna(subset=['content'])

# Tokenize the content and store it in a new column
data['tokens'] = data['content'].apply(tokenize_text)

print("Done tokenizing")

# Convert tokens to a string
data['tokens_str'] = data['tokens'].apply(lambda tokens: ' '.join(tokens))

# # Create a TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data['tokens_str'])

X = data['tokens_str']
labels = data['spam']

# Trains a logistic regression model using 10-fold cross-validation
nSplit = 5
kfold = KFold(n_splits=nSplit, shuffle=True, random_state=42)

accuracyScores = []
f1Scores = []
precisionScores = []
recallScores = []

i = 0
for train, test in kfold.split(X):
    print(f"testing fold {i}")
    i += 1
    trainX, testX = X[train], X[test]
    trainY, testY = labels.iloc[train], labels.iloc[test]

    model = LogisticRegression(max_iter=5000, random_state=42)
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

# save the model locally
pickle.dump(model, open('linkContentClassifier.sav', 'wb'))
