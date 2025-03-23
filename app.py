from flask import Flask, request, render_template
from text_cleaner import preprocess_text
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/logreg_model.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_tokens = preprocess_text(text)
        cleaned_text = ' '.join(cleaned_tokens)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        return render_template('result.html', 
                             cleaned_text=cleaned_text,
                             prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)