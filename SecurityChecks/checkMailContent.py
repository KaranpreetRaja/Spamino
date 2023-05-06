import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import DistilBertTokenizer, DistilBertModel

class BinaryClassifier(nn.Module):
    def __init__(self, distilbert_model):
        super(BinaryClassifier, self).__init__()
        self.distilbert = distilbert_model
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.distilbert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits.squeeze(-1)
    
    
def checkContent():
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    classifier = BinaryClassifier(distilbert_model)
    classifier.load_state_dict(torch.load("binary_classifier.pt"))
    classifier.eval()
    new_email_title = "Your new email title here"
    new_email_body = "Your new email body here"

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    max_len = tokenizer.max_model_input_sizes['distilbert-base-uncased']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title_encodings = tokenizer(new_email_title, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    body_encodings = tokenizer(new_email_body, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")

    title_input_ids = title_encodings["input_ids"].to(device)
    title_attention_mask = title_encodings["attention_mask"].to(device)
    body_input_ids = body_encodings["input_ids"].to(device)
    body_attention_mask = body_encodings["attention_mask"].to(device)

    with torch.no_grad():
        title_outputs = classifier(title_input_ids, title_attention_mask)
        body_outputs = classifier(body_input_ids, body_attention_mask)
        outputs = (title_outputs + body_outputs) / 2
        prediction = torch.round(torch.sigmoid(outputs)).item()

    return prediction



