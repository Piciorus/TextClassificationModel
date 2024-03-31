import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch import nn
from config import DEVICE, DATA_FILE_PATH, NUM_EPOCHS, LEARNING_RATE, CHUNK_SIZE, VECTORIZER, HIDDEN_SIZE, NUM_CLASSES
from model import TextClassifier


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def train_model(data_file_path, vectorizer, model, device, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    chunk_size = 10000

    for epoch in range(num_epochs):
        data_stream = pd.read_csv(data_file_path, chunksize=chunk_size)
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        for data_chunk in data_stream:
            data_chunk = data_chunk.dropna()
            X_chunk = vectorizer.transform(data_chunk['Text'])
            y_chunk = data_chunk['Label']
            y_chunk_tensor = torch.tensor(y_chunk.values - 1, dtype=torch.long)
            X_chunk_tensor = torch.tensor(X_chunk.toarray(), dtype=torch.float32).to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(X_chunk_tensor)
            loss = criterion(outputs, y_chunk_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_accuracy += calculate_accuracy(y_chunk_tensor.cpu().numpy(), predicted.cpu().numpy())
            num_batches += 1

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}')


        save_checkpoint(model, optimizer, epoch, average_loss, f"checkpointbetter{epoch}.pt")

    print("Training finished.")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = pd.read_csv(DATA_FILE_PATH, nrows=CHUNK_SIZE)
    X_chunk = VECTORIZER.fit_transform(chunk['Text'])
    input_size = X_chunk.shape[1]
    model = TextClassifier(input_size, HIDDEN_SIZE, NUM_CLASSES).to(device)
    train_model(DATA_FILE_PATH, VECTORIZER, model, DEVICE, NUM_EPOCHS, LEARNING_RATE)

