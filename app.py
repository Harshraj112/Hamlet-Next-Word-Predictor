
import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Use st.cache_resource to load model and tokenizer
@st.cache_resource
def load_my_model():
    st.write("Loading model...")
    model = load_model('next_word_lstm.h5')
    st.write("Model loaded")
    return model

@st.cache_resource
def load_tokenizer():
    st.write("Loading tokenizer...")
    with open('tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    st.write("Tokenizer loaded")
    return tokenizer

model = load_my_model()
tokenizer = load_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    st.write(f"Input text: {text}")
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    st.write(f"Raw model output: {predicted}")
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")

# Use session state safely
if "result" not in st.session_state:
    st.session_state.result = None

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.session_state.result = next_word
    st.write(f'Next word: {next_word}')

