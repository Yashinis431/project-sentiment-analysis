import pickle

# Load vectorizer and model
with open("vectorizer.pkl", "rb") as vec_file:  # ✅ Correct file extension
    vectorizer = pickle.load(vec_file)  # ✅ Correct loading method

with open("model.pkl", "rb") as model_file:  # ✅ Correct file extension
    model = pickle.load(model_file)  # ✅ Correct loading method

def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])  # ✅ No error
    prediction = model.predict(review_tfidf)  # ✅ Use trained model
    return prediction[0]  # Return single prediction

print(predict_sentiment("Great movie!"))
