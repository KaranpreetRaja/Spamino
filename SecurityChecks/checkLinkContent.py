import pickle
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


#use model to classify a new link
def classify_link(link):
    model = pickle.load(open('linkContentClassifier.sav', 'rb'))
    return model.predict(tokenize_text(get_content(link)))
