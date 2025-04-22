import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from preprocess import clean_text

# Load data
df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['message'] = df['message'].apply(clean_text)

# Prepare data
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
