import joblib
from src.preprocess import clean_text

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_spam(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"
