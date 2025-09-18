from flask import Flask, render_template, request
import pickle
import re

# Load the pipeline (vectorizer + model together)
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['user_input']
        cleaned = clean_text(input_text)

        # Just call predict on pipeline
        prediction = model.predict([cleaned])[0]

        result = "Hate Speech ❌" if prediction == 1 else "Not Hate Speech ✅"
        return render_template("index.html",
                               prediction_text=f"Input: {input_text} → {result}")

if __name__ == "__main__":
    app.run(debug=True)
