# 📧 Spam Message Classifier

This project uses **Machine Learning** to classify SMS messages as **Spam** or **Not Spam** using **Python**, **Scikit-learn**, and **NLTK**.

---

## 🚀 Features

- Preprocesses and cleans raw SMS data.
- Trains a Naive Bayes classifier with TF-IDF vectorization.
- Predicts user input messages as Spam or Not Spam.
- 96%+ accuracy on test data.

---

## 🧠 Algorithms & Libraries

- **Model**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF
- **Libraries**: `sklearn`, `pandas`, `nltk`, `joblib`


---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Varshneyhars/Spam-Message-Classifier.git
   cd spam-message-classifier
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download dataset Place spam.csv inside the data/ folder. Download dataset from Kaggle

🏃‍♂️ How to Run
1. Train the model:
bash
Copy
Edit
python src/train_model.py
2. Test a message:
bash
Copy
Edit
python app.py
Example:
less
Copy
Edit
Enter a message (or type 'exit'): You have won a free iPhone!
Prediction: Spam
✅ Accuracy
Achieved 96% accuracy on test data split using 80/20.

📌 To-Do
 Add GUI or web interface (Flask/Streamlit)

 Deploy to HuggingFace or Heroku

 Add confusion matrix visualization

📚 License
MIT License - free to use and modify for personal or commercial use.

🤝 Contributing
PRs are welcome! Feel free to fork and enhance the project.

