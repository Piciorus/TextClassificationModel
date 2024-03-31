import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from config import DEVICE, CLASS_LABELS, CHECKPOINT_PATH, DATA_FILE_PATH, CHUNK_SIZE, HIDDEN_SIZE, NUM_CLASSES, \
    CLASS_LABELS_MATRIX
from test_data import question_per_class
from utils import load_checkpoint


def calculate_accuracy(predictions, ground_truth):
    correct = 0
    total = len(predictions)
    for pred, true_label in zip(predictions, ground_truth):
        if pred == true_label:
            correct += 1
    accuracy = correct / total

    return accuracy

def predict_sentences_and_calculate_accuracy(model, sentences_per_class, vectorizer, class_labels, device):
    predicted_labels = []
    true_labels = []
    for category, sentences in sentences_per_class.items():
        for sentence in sentences:
            text_to_predict_vectorized = vectorizer.transform([sentence])
            with torch.no_grad():
                outputs = model(torch.tensor(text_to_predict_vectorized.toarray(), dtype=torch.float32).to(device))
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item() + 1
            predicted_class = class_labels[predicted_class_index]
            predicted_labels.append(predicted_class)
            true_labels.append(category)
    accuracy = calculate_accuracy(predicted_labels, true_labels)

    return accuracy, true_labels, predicted_labels

def predict_category(model, sentence, vectorizer, class_labels, device):
    text_to_predict_vectorized = vectorizer.transform([sentence])
    with torch.no_grad():
        outputs = model(torch.tensor(text_to_predict_vectorized.toarray(), dtype=torch.float32).to(device))
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item() + 1
    predicted_category = class_labels[predicted_class_index]
    return predicted_category

if __name__ == '__main__':
    checkpoint_filepath = CHECKPOINT_PATH
    chunk = pd.read_csv(DATA_FILE_PATH, nrows=CHUNK_SIZE)
    vectorizer = CountVectorizer()
    X_chunk = vectorizer.fit_transform(chunk['Text'])
    input_size = X_chunk.shape[1]

    model = load_checkpoint(CHECKPOINT_PATH, input_size, HIDDEN_SIZE, NUM_CLASSES)

    ground_truth_labels = chunk['Label'].tolist()

    accuracies = []
    for category, sentences in question_per_class.items():
        print(f"---Questions from category : {category}")
        for sentence in sentences:
            accuracy, true_labels, predicted_labels = predict_sentences_and_calculate_accuracy(model, question_per_class, vectorizer, CLASS_LABELS, DEVICE)
            accuracies.append(accuracy)
            predicted_category = predict_category(model, sentence, vectorizer, CLASS_LABELS, DEVICE)
            print(f"Question: {sentence} - Predicted Category: {predicted_category}")

    print(f"Average Accuracy model: {np.mean(accuracies)}")

    cm = confusion_matrix(true_labels, predicted_labels, labels=CLASS_LABELS_MATRIX)

    report = classification_report(true_labels, predicted_labels, target_names=CLASS_LABELS_MATRIX)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix\nModel Accuracy: {np.mean(accuracies):.2f}')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_LABELS_MATRIX))
    plt.xticks(tick_marks, CLASS_LABELS_MATRIX, rotation=45)
    plt.yticks(tick_marks, CLASS_LABELS_MATRIX)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

