from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1. Sample text data to fit vectorizer on
texts = [
    "I love this movie",
    "This movie was terrible",
    "Great acting and story",
    "Worst movie I have seen",
    "Amazing direction and cast"
]

# 2. Create the vectorizer
vectorizer = TfidfVectorizer()

# 3. Fit the vectorizer on the text data
vectorizer.fit(texts)

# 4. Save the vectorizer to a file named 'vectorizer.pkl'
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("vectorizer.pkl saved successfully!")
