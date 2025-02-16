import streamlit as st
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

# Loading pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Loading the saved model and TF-IDF vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
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
        text_tfidf = tfidf.transform([cleaned_text])
        # Predict the sentiment
        prediction = model.predict(text_tfidf)[0]
        # Display the result
        st.write(f'Sentiment: {prediction}')
    else:
        st.write('Please enter a review to analyze.')

# CSV Upload Section
st.subheader("Upload a CSV File")
uploaded_file = st.file_uploader("Upload a CSV file containing reviews", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        st.write("Processing CSV file...")
        df['cleaned_review'] = df['review'].astype(str).apply(preprocess_text)
        df['vector'] = df['cleaned_review'].apply(lambda x: text_to_vector(x, word2vec_model))
        df['sentiment'] = model.predict(np.vstack(df['vector']))
        st.write(df[['review', 'sentiment']])
        
        # Download option
        csv = df[['review', 'sentiment']].to_csv(index=False)
        st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")
    else:
        st.write("The uploaded CSV file must contain a 'review' column.")



