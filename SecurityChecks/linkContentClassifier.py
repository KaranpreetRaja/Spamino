import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoTokenizer


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

