# pip section :)

!pip install -q transformers datasets gradio

# imports section

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
import gradio as gr
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# setup dataset
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")


# setup tokenizer and dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = load_dataset("imdb")  

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Roberta + BiLSTM + Attention setup
class RobertaBiLSTMAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(roberta_out)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        output = self.fc(self.dropout(context))
        return output


# setup train set and batch size
from torch.utils.data import DataLoader

train_loader = DataLoader(encoded_dataset["train"].select(range(20000)), batch_size=16, shuffle=True)
test_loader = DataLoader(encoded_dataset["test"].select(range(2000)), batch_size=16)



# training process
from tqdm import tqdm

model = RobertaBiLSTMAttention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(10):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n if progress_bar.n else 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")



# inner evaluation
model.eval()

samples = [
    # 1: sarcasm
    "Wow, what a masterpiece. I especially loved the part where nothing happened for two hours.",  # 🔴 Negative

    # 2: emotionally mixed
    "The movie was boring at times, but the ending completely blew my mind.",  # 🟢 Positive (نسبتاً مثبت)

    # 3: ta’arofy tone (fake praise)
    "It was... fine, I guess. Not bad. Not good. Just there.",  # 🔴 Negative (neutral leaning negative)

    # 4: visually good but poor content
    "Beautiful cinematography can’t save a script written by a potato.",  # 🔴 Negative

    # 5: praise with weird slang
    "Yo that movie was sick af! 🔥🔥",  # 🟢 Positive ("sick" = slang for amazing)

    # 6: backhanded compliment
    "I didn’t expect much, and yet it still managed to disappoint me.",  # 🔴 Negative

    # 7: full sarcasm
    "10/10 would recommend... if you enjoy falling asleep halfway through.",  # 🔴 Negative

    # 8: fake excitement
    "Absolutely incredible! I only checked my phone 12 times.",  # 🔴 Negative

    # 9: nostalgic + honest
    "Reminded me of my childhood, cheesy but heartwarming.",  # 🟢 Positive

    # 10: hype tone
    "Bro that film went HARD. Straight banger!",  # 🟢 Positive (slang-heavy positive)
]



for s in samples:
    tokens = tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()
        print(f"{s} ➤ {'🟢 Positive' if pred == 1 else '🔴 Negative'} ")




# UI w/ gradio
def predict(text):
    model.eval()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item()
    label = "🟢 Positive" if pred == 1 else "🔴 Negative"
    return f"{label} ({conf*100:.1f}%)"

gr.Interface(fn=predict, inputs=gr.Textbox(label="Enter a review"), outputs="text").launch()




