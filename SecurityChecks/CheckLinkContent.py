import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from requests.exceptions import Timeout
from bs4 import BeautifulSoup

def InitializeVectorizerAndModel():
    objects = []
    with (open("linkContentClassifier.sav", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    model = objects[0]

    data = pd.read_csv("/root/Spamino/SecurityChecks/urlCont.csv")

    X = data['content'].values
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=1000)
    vectorizer.fit(data['content'].values)
    return vectorizer, model

def checkLinkContent(link, vectorizer, model):
    try:
        response = requests.get(link, timeout=2)
        if response.status_code == 200:
            soupContent = BeautifulSoup(response.content, 'html.parser')

            str =' '.join([text.get_text() for text in soupContent.find_all(['h1', 'h2', 'h3', 'p'])])
            if str =="":
                return False
            X_test = vectorizer.transform([str])
            try:
                prediction = model.predict(X_test)
                return prediction
            except ValueError:
                print("ValueError: X has 1000 features, but LogisticRegression is expecting 3 features as input.")
                return False
        else:
            return False

    except Timeout:
        return False
