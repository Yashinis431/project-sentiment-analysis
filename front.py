import streamlit as st
from textblob import TextBlob

st.title("🧠 Sentiment Analysis App")
st.write("Enter a sentence or paragraph below:")

text_input = st.text_area("Your Text")

if st.button("Analyze"):
    if text_input:
        blob = TextBlob(text_input)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            st.success("Positive 😊")
        elif sentiment < 0:
            st.error("Negative 😠")
        else:
            st.info("Neutral 😐")
