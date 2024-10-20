from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = load_model('tfidf_ann_model.h5')

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Transform the input text using the TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([text]).toarray()
    # Predict using the trained model
    prediction = model.predict(transformed_text)
    # Convert probabilities to binary labels
    label = 1 if prediction >= 0.5 else 0
    result = 'Spam' if label == 1 else 'Ham'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
