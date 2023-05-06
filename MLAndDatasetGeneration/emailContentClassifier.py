import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel



class EmailDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        row = self.data.iloc[index]
        title = row["title"]
        body = row["body"]
        tld = row["tld"]
        has_return_path = row["has_return_path"]
        x_bulkmail = row["x_bulkmail"]
        label = row["label"]

        # Tokenize and create attention mask for title and body
        title_encodings = self.tokenizer(title, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        body_encodings = self.tokenizer(body, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")

        # print(f"title: {title}")
        # print(f"title_encodings['input_ids'].shape: {title_encodings['input_ids'].shape}")
        # print(f"title_encodings['attention_mask'].shape: {title_encodings['attention_mask'].shape}")
        # print(f"body: {body}")
        # print(f"body_encodings['input_ids'].shape: {body_encodings['input_ids'].shape}")

        return {
            "title_input_ids": title_encodings["input_ids"].squeeze(0),
            "title_attention_mask": title_encodings["attention_mask"].squeeze(0),
            "body_input_ids": body_encodings["input_ids"].squeeze(0),
            "body_attention_mask": body_encodings["attention_mask"].squeeze(0),
            "tld": torch.tensor(tld, dtype=torch.long),
            "has_return_path": torch.tensor(has_return_path, dtype=torch.float32),
            "x_bulkmail": torch.tensor(x_bulkmail, dtype=torch.float32),
            "targets": torch.tensor(label, dtype=torch.float32),
        }



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


# tokenizes the data and Truncate and add padding to the data
def tokenize_and_truncate(string, tokenizer, max_len):
    tokens = tokenizer.tokenize(string)
    tokens = tokens[:max_len-2]
    # Add padding tokens
    tokens += ['[PAD]'] * (max_len - 2 - len(tokens))
    return " ".join(tokens)



def load_data(tokenizer):
    # Read the CSV file
    data = pd.read_csv("exdata.csv")


    # get tokenizer max length
    max_len = tokenizer.max_model_input_sizes['distilbert-base-uncased']



    print("Number of rows: ", len(data))

    # Preprocess the data
    # 1. Title of the email (tokenize using BERT)
    data["title"] = data["title"].fillna("")
    # use tokenize_and_truncate function to tokenize and truncate the data
    data["title"] = data["title"].apply(lambda x: tokenize_and_truncate(x, tokenizer, max_len))


    # 2. Body/Text of the email (tokenize using BERT)
    data["body"] = data["body"].fillna("")
    data["body"] = data["body"].apply(lambda x: tokenize_and_truncate(x, tokenizer, max_len))

    # 3. TLD
    data["tld"] = data["tld"].astype('category').cat.codes

    # 4. If has Return Path (has value "True" or "False")
    data["has_return_path"] = data["return_path"].notna().astype(np.float32)

    # Print number of times has_return_path is 1
    print(data["has_return_path"].value_counts())

    # 5. X-bulkmail (if has one then write the float, otherwise 0)
    data["x_bulkmail"] = data["x_bulkmail"].fillna(0).astype(np.float32)

    # 6. X-keywords (give me a comma separated list of all x-keywords, if doesn't exist, then empty)
    data["x_keywords"] = data["x_keywords"].fillna("").str.split(",")

    print(data["label"])

    # 7. label (Yes if spam, No if not-spam) (convert to 1 and 0)
    data["label"] = data["label"].replace({"Yes": 1, "No": 0})

    # Define the features and target
    features = ["title", "body", "tld", "has_return_path", "x_bulkmail"]

    print(data.dtypes)


    target = "label"

    X = data[features].values
    y = data[target].values

    print(data["label"])

    return data


# Chooses the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print we are using GPU if available
if torch.cuda.is_available():
    print("Using CUDA!")



# Loads the data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
max_len = tokenizer.max_model_input_sizes['distilbert-base-uncased']
data = load_data(tokenizer)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = EmailDataset(train_data, tokenizer, max_len)
test_dataset = EmailDataset(test_data, tokenizer, max_len)


# Instantiate the BERT model 
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Instantiate the classifier
classifier = BinaryClassifier(distilbert_model).to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=2e-5)

# Define the data loader and train the model
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
num_epochs = 5

count = 0
torch.cuda.empty_cache()
# Training loop go BRRRRRRRRR
accumulation_steps = 4
optimizer.zero_grad()
for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    for batch in train_loader:
        count += 1
        print("Batch: ", count)
        title_input_ids = batch["title_input_ids"].to(device)
        title_attention_mask = batch["title_attention_mask"].to(device)
        body_input_ids = batch["body_input_ids"].to(device)
        body_attention_mask = batch["body_attention_mask"].to(device)
        targets = batch["targets"].to(device)
        title_outputs = classifier(title_input_ids, title_attention_mask)
        body_outputs = classifier(body_input_ids, body_attention_mask)
        outputs = (title_outputs + body_outputs) / 2
        loss = criterion(outputs, targets)
        loss.backward()
        if (count % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()


# Evaluate the trained model on a test set
y_pred = []
y_test = []
for batch in test_loader:
    title_input_ids = batch["title_input_ids"].to(device)
    title_attention_mask = batch["title_attention_mask"].to(device)
    body_input_ids = batch["body_input_ids"].to(device)
    body_attention_mask = batch["body_attention_mask"].to(device)
    targets = batch["targets"].to(device)
    with torch.no_grad():
        title_outputs = classifier(title_input_ids, title_attention_mask)
        body_outputs = classifier(body_input_ids, body_attention_mask)
        outputs = (title_outputs + body_outputs) / 2
        outputs = torch.sigmoid(outputs)
        y_pred.extend(torch.round(outputs).tolist())
        y_test.extend(targets.tolist())



#prints f1 score, precision, recall, accuracy
print("F1 score: {:.2f}".format(f1_score(y_test, y_pred)))
print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))


# Saves trained models
torch.save(classifier.state_dict(), "binary_classifier.pt")
