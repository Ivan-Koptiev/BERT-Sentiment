import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
    def prepare_data(self, texts, labels=None, max_length=128):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        if labels is not None:
            encodings['labels'] = torch.tensor(labels)
        return encodings
    def train(self, train_texts, train_labels, val_texts, val_labels,
              batch_size=16, epochs=3, learning_rate=2e-5):
        train_encodings = self.prepare_data(train_texts, train_labels)
        val_encodings = self.prepare_data(val_texts, val_labels)
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_encodings['labels']
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_texts, val_labels)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    def evaluate(self, texts, labels):
        self.model.eval()
        encodings = self.prepare_data(texts)
        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch_input_ids = encodings['input_ids'][i:i+16].to(self.device)
                batch_attention_mask = encodings['attention_mask'][i:i+16].to(self.device)
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                batch_predictions = torch.argmax(outputs.logits, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
        return accuracy_score(labels, predictions)
    def predict(self, texts):
        self.model.eval()
        encodings = self.prepare_data(texts)
        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch_input_ids = encodings['input_ids'][i:i+16].to(self.device)
                batch_attention_mask = encodings['attention_mask'][i:i+16].to(self.device)
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                batch_predictions = torch.argmax(outputs.logits, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
        return predictions

def train_text_classifier():
    # Load the sentiment data with correct column names
    df = pd.read_csv('/home/ivan-koptiev/Codes/Codes/portfolio website/github projects/NLP_Bert/sentiment_data.csv')
    texts = df['SentimentText'].astype(str).tolist()
    labels = df['Sentiment'].astype(int).tolist()
    # Use 20,000 for training, 5,000 for validation
    train_texts = texts[:20000]
    train_labels = labels[:20000]
    val_texts = texts[20000:25000]
    val_labels = labels[20000:25000]
    classifier = BERTClassifier(num_labels=2)
    # Training parameters
    batch_size = 32
    epochs = 7
    learning_rate = 1e-5
    best_val_acc = 0
    patience = 2
    patience_counter = 0
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        classifier.train(train_texts, train_labels, val_texts, val_labels, batch_size=batch_size, epochs=1, learning_rate=learning_rate)
        val_acc = classifier.evaluate(val_texts, val_labels)
        print(f"Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(classifier.model.state_dict(), 'best_bert_sentiment.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping: no improvement in validation accuracy.")
                break
    # Load best model
    classifier.model.load_state_dict(torch.load('best_bert_sentiment.pt'))
    # Validation set evaluation: classification report and confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    val_preds = classifier.predict(val_texts)
    report = classification_report(val_labels, val_preds, target_names=['Negative', 'Positive'])
    print("\nValidation Classification Report:\n", report)
    cm = confusion_matrix(val_labels, val_preds)
    # Visualize confusion matrix
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0,1], ['Negative', 'Positive'])
    plt.yticks([0,1], ['Negative', 'Positive'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
    plt.tight_layout()
    plt.savefig('val_confusion_matrix.png')
    plt.close()
    # Show a few misclassified examples
    print("\nSome misclassified validation examples:")
    shown = 0
    for text, true, pred in zip(val_texts, val_labels, val_preds):
        if true != pred:
            print(f"Text: {text}\nTrue: {'Positive' if true==1 else 'Negative'}, Predicted: {'Positive' if pred==1 else 'Negative'}\n")
            shown += 1
            if shown >= 5:
                break
    # Demo: Predict on a larger, diverse set of new texts
    demo_texts = [
        "I love this movie! It was fantastic.",
        "This is the worst product I have ever bought.",
        "Not bad, but could be better.",
        "Absolutely wonderful experience.",
        "I hate waiting in line.",
        "The food was okay, nothing special.",
        "Best day ever!",
        "Terrible customer service.",
        "I'm not sure how I feel about this.",
        "It was fine, I guess.",
        "I can't recommend this enough!",
        "Awful, just awful.",
        "The staff was friendly and helpful.",
        "I will never come back here again.",
        "Mediocre at best.",
        "Exceeded my expectations!",
        "The packaging was damaged.",
        "Superb quality and fast shipping.",
        "Disappointing experience overall.",
        "I am so happy with my purchase!",
        "It broke after one use.",
        "Five stars!",
        "One of the best I've tried.",
        "Not worth the money.",
        "I feel neutral about this.",
        "It was okay, not great, not terrible.",
        "I expected more for the price.",
        "My friends all loved it.",
        "Would not buy again.",
        "A pleasant surprise!"
    ]
    demo_preds = classifier.predict(demo_texts)
    print("\nDemo predictions:")
    for text, pred in zip(demo_texts, demo_preds):
        label = 'Positive' if pred == 1 else 'Negative'
        print(f"Text: {text}\nPredicted Sentiment: {label}\n")
    # Visualization: Bar plot of predicted sentiment counts
    plt.figure(figsize=(5, 3))
    labels_map = {0: 'Negative', 1: 'Positive'}
    pred_counts = [demo_preds.count(0), demo_preds.count(1)]
    plt.bar(['Negative', 'Positive'], pred_counts, color=['red', 'green'])
    plt.title('Predicted Sentiment Counts (Demo Texts)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('demo_sentiment_counts.png')
    plt.close()
    # Visualization: Table of demo texts and predictions
    fig, ax = plt.subplots(figsize=(10, 5))
    table_data = [[t, labels_map[p]] for t, p in zip(demo_texts, demo_preds)]
    table = ax.table(cellText=table_data, colLabels=["Text", "Predicted Sentiment"], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('demo_sentiment_table.png')
    plt.close()

if __name__ == "__main__":
    train_text_classifier()