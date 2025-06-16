from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from data_preprocess import clean_text

# Sample training data
texts = [
    "I love this product",
    "Worst experience ever",
    "Very happy with service",
    "Not good at all",
    "Excellent support",
    "Bad quality and rude staff"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Clean the texts
cleaned_texts = [clean_text(text) for text in texts]

# Create and fit TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X, labels)

# Save the vectorizer and model together as a tuple
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model saved successfully as model.pkl")
