import os

import torch
import pandas as pd

from flask import Flask, request, jsonify
from sklearn.exceptions import NotFittedError
from config import DATA_FILE_PATH, CHUNK_SIZE, DEVICE, VECTORIZER, HIDDEN_SIZE, NUM_CLASSES, CLASS_LABELS, \
    CHECKPOINT_PATH, API_URL
from utils import load_checkpoint
from openai import OpenAI
from flask_cors import CORS

api_key = os.getenv("API_KEY")

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=[API_URL])

client = OpenAI(api_key=api_key)

def predict_category(question_text, model, vectorizer, class_labels, device):
    try:
        vectorizer.vocabulary_
    except AttributeError:
        raise NotFittedError("Vocabulary not fitted or provided")

    text_to_predict_vectorized = vectorizer.transform([question_text])
    with torch.no_grad():
        outputs = model(torch.tensor(text_to_predict_vectorized.toarray(), dtype=torch.float32).to(device))
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item() + 1
    predicted_class = class_labels[predicted_class_index]
    return predicted_class


@app.route('/predict_category', methods=['POST'])
def predict_category_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        question_text = data['question']

        predicted_category = predict_category(question_text, model, VECTORIZER, CLASS_LABELS, DEVICE)

        return jsonify({'predicted_category': predicted_category})

@app.route('/get_correct_answer', methods=['POST'])
def get_correct_answer():
    data = request.json
    question_description = data.get('question_description')
    answers = data.get('answers')

    prompt = f"Question: {question_description}\n\nAnswers:\n {answers[0]}\n {answers[1]}\n {answers[2]}\n\nChoose the correct answer.Provide me only answer without any other words."

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return jsonify({'correct_answer': completion.choices[0].message.content})

if __name__ == '__main__':
    chunk = pd.read_csv(DATA_FILE_PATH, nrows=CHUNK_SIZE)
    X_chunk = VECTORIZER.fit_transform(chunk['Text'])
    input_size = X_chunk.shape[1]
    model = load_checkpoint(CHECKPOINT_PATH, input_size, HIDDEN_SIZE, NUM_CLASSES)
    app.run(debug=True)
