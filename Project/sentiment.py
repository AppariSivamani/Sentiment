from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

app = Flask(__name__, static_folder='static')


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def classify_feedback(feedback):
    inputs = tokenizer(feedback, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    sentiment_labels = {0: "Very Bad", 1: "Bad", 2: "Neutral", 3: "Good", 4: "Very Good"}
    return sentiment_labels.get(predicted_class, "unknown")

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", sentiment="")

@app.route('/predict', methods=['POST'])
def predict():
    doctor_name = request.form.get("doctor_name", "")
    feedback = request.form.get("feedback", "")
    sentiment = classify_feedback(feedback)
    return render_template("index.html", sentiment=sentiment)

if __name__ == '__main__':
    feedback = "The doctor was good."
    print("Sentiment:", classify_feedback(feedback))
    app.run(debug=True)
