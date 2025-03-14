from flask import Flask, request, jsonify
from sentiment_model import classify_text

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_feedback():
    data = request.get_json()
    feedback = data.get("feedback", "")
    
    if not feedback:
        return jsonify({"error": "No feedback provided"}), 400
    
    sentiment = classify_text(feedback)
    return jsonify({"feedback": feedback, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
