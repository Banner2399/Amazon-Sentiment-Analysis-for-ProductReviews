import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import gensim.downloader as api


# Downloading NLTK data
nltk.download('stopwords')
nltk.download('wordnet')



# Precompute stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Load the trained model
model = joblib.load('models/sentiment_model.pkl')


# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Convert text to Word2Vec vector
def text_to_vector(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Streamlit app
st.title('Amazon Product Review Sentiment Analysis')


st.write('Enter a product review to predict its sentiment.')

# User input
user_input = st.text_area("Review Input", "")

if st.button("Analyze Sentiment"): 
    if user_input:
        #preprocess the input
        cleaned_text = preprocess_text(user_input) 
        # Vectorize the input
        text_vector = text_to_vector(cleaned_text, word2vec_model)
        # Predict the sentiment
        prediction = model.predict(text_vector)[0]
        # Display the result
        st.write(f'Sentiment: {prediction}')
    else:
        st.write('Please enter a review to analyze.')

#

