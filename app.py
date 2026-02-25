from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 🔹 NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# 🔹 Preprocess
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stpwrds]
    return " ".join(words)

# 🔹 Train model if not exists
def train_model():
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    true_df["label"] = 0
    fake_df["label"] = 1

    data = pd.concat([true_df, fake_df]).sample(frac=1)

    data["text"] = data["text"].apply(preprocess)

    vector = TfidfVectorizer(max_features=5000)
    X = vector.fit_transform(data["text"])
    y = data["label"]

    model = PassiveAggressiveClassifier()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vector, open("vector.pkl", "wb"))

    return model, vector

# 🔹 Load or train
if os.path.exists("model.pkl") and os.path.exists("vector.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
    vector = pickle.load(open("vector.pkl", "rb"))
else:
    print("Training model first time...")
    model, vector = train_model()

# 🔹 Prediction
def fake_news_det(news):
    processed = preprocess(news)
    vect = vector.transform([processed])
    pred = model.predict(vect)
    return pred

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('news')

        if not message:
            return render_template("prediction.html", prediction_text="No input provided")

        pred = fake_news_det(message)

        if pred[0] == 1:
            result = "Prediction: Fake News 📰"
        else:
            result = "Prediction: Real News 📰"

        return render_template("prediction.html", prediction_text=result)

    return render_template("prediction.html")

# 🔹 Run
if __name__ == '__main__':
    app.run(debug=True)