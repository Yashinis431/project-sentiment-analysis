import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load dataset
file_path = "C:/project/IMDB Dataset.csv"  # Adjust path if needed
df = pd.read_csv(file_path)  # Read dataset
reviews = df['review'].astype(str)  # Convert to string (avoid NaN issues)

# Step 2: Initialize and Fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()  # ✅ Now vectorizer is defined
X_tfidf = vectorizer.fit_transform(reviews)  # ✅ Now vectorizer is trained

# Step 3: Save the trained vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Vectorizer has been saved successfully!")

