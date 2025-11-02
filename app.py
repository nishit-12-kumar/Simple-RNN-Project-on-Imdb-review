import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Step 1: Load IMDB word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2) + 3
        if index >= 10000:
            index = 2
        encoded_review.append(index)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def is_english(text):
    """Check if the text contains only English characters."""
    return re.match("^[a-zA-Z0-9\s.,!?']*$", text) is not None



# Step 3: Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning("Please enter a review before classifying.")
    elif not is_english(user_input):
        st.warning("Please enter text in English only. Hindi words are not supported.")

    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        if sentiment =='Positive':
            st.success(f'**Sentiment:** {sentiment}')
        else:
            st.error(f'**Sentiment:** {sentiment}')
        
        st.info(f'**Prediction Score:** {prediction[0][0]:.4f}')

else:
    st.write('Please enter a movie review.')
