import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")  # Ensure this dataset exists

# Preprocess text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"].map({"positive": 1, "negative": 0})

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully!")
