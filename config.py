import torch
from sklearn.feature_extraction.text import CountVectorizer

DATA_FILE_PATH = "dataset/preprocess_dataset.csv"
CHECKPOINT_PATH = "checkpoints/checkpointbetter10.pt"
CHUNK_SIZE = 10000
HIDDEN_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 11
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VECTORIZER = CountVectorizer()
API_URL = 'http://localhost:4200'

CLASS_LABELS = {
    1: "Society & Culture",
    2: "Science & Mathematics",
    3: "Health",
    4: "Education & Reference",
    5: "Computers & Internet",
    6: "Sports",
    7: "Business & Finance",
    8: "Entertainment & Music",
    9: "Family & Relationships",
    10: "Politics & Government",
}

CLASS_LABELS_MATRIX = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government"
]
