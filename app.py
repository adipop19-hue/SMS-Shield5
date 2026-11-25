
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved ML model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form.get("message", "")

    # Convert SMS into TF-IDF format
    message_tfidf = vectorizer.transform([message])

    # Predict spam or not spam
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=result,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)





