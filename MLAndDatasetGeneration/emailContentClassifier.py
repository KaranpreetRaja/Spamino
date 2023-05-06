import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Defines custom dataset for PyTorch
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_sequence_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_features = self.tokenizer(row['title'] + row['body'], padding='max_length', truncation=True, return_tensors='pt')
        actual_sequence_length = text_features['input_ids'].size(1)
        padding_length = self.max_sequence_length - actual_sequence_length
        if padding_length < 0:
            padding_length = 0
        elif padding_length >= self.max_sequence_length:
            padding_length = 0
        text_features_padded = {key: torch.nn.functional.pad(value[0], (0, 0, 0, padding_length)) if padding_length > 0 else value[0] for key, value in text_features.items()}
        numerical_features = torch.tensor([row['tld'], row['has_return_path'], row['x_bulkmail']], dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return text_features_padded, numerical_features, label


# Defines binary classifier using PyTorch
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Defines Deep Q-Network (DQN) agent w/ PyTorch
class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, targets = [], []
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                next_q_values = self.model(next_state)
                target = reward + self.gamma * torch.max(next_q_values)
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            q_values[action] = target
            states.append(state)
            targets.append(q_values)
        states = torch.stack(states)
        targets = torch.stack(targets)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(1):
            optimizer.zero_grad()
            outputs = self.model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# tokenizes the data and Truncate
def tokenize_and_truncate(string, tokenizer, max_len):
    tokens = tokenizer.tokenize(string)
    tokens = tokens[:max_len-2]
    return tokens


def load_data(tokenizer):
    # Read the CSV file
    data = pd.read_csv("exdata.csv")


    # get tokenizer max length
    max_len = tokenizer.max_model_input_sizes['bert-base-uncased']



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

    # 7. label (Yes if spam, No if not-spam)
    data["label"] = data["label"].map({"Yes": 1, "No": 0}).astype(np.float32)

    # Define the features and target
    features = ["title", "body", "tld", "has_return_path", "x_bulkmail"]

    print(data.dtypes)


    target = "label"

    # Create X and y
    dataset = CustomDataset(data, tokenizer, max_len)

    return dataset

# Chooses the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print we are using GPU if available
if torch.cuda.is_available():
    print("Using CUDA!")

# Loads the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
dataset = load_data(tokenizer)

# Splits the data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Creates the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# Creates the binary classifier and trains it using PyTorch lib
clf = BinaryClassifier(input_size=3)
criterion = nn.BCELoss()
optimizer = optim.Adam(clf.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        text_features, numerical_features, labels = batch
        optimizer.zero_grad()
        outputs = clf(numerical_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Epoch {}/{} - loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# # Creates DQN agent and train it using PyTorch
# state_size = X_train.shape[1]
# action_size = 2
# agent = DQNAgent(state_size, action_size)
# num_episodes = 1000
# for episode in range(num_episodes):
#     state = X_train[0]
#     done = False
#     while not done:
#         action = agent.act(state)
#         next_state = X_train[action]
#         reward = y_train[action]
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         agent.replay()
#     agent.decay_epsilon()
#     if (episode+1) % 100 == 0:
#         print("Episode {}/{}".format(episode+1, num_episodes))

# Evaluates the binary classifier
y_test = []
y_pred = []
clf.eval()
with torch.no_grad():
    for batch in test_loader:
        text_features, numerical_features, labels = batch
        outputs = clf(numerical_features)
        y_test.extend(labels.numpy())
        y_pred.extend(np.where(outputs.numpy() > 0.5, 1, 0))

y_test = np.array(y_test)
y_pred = np.array(y_pred)

#prints f1 score, precision, recall, accuracy
print("F1 score: {:.2f}".format(f1_score(y_test, y_pred)))
print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))


# Saves trained models
clf.save("binary_classifier.pt")
# agent.save("dqn_agent.pt")
