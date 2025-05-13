import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert text data to numerical data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer


