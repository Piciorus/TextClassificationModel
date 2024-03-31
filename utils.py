import torch

from model import TextClassifier


def load_checkpoint(filepath, input_size, hidden_size, num_classes):
    model = TextClassifier(input_size, hidden_size, num_classes)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
