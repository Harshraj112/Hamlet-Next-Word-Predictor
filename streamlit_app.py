import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = tf.keras.models.load_model('next_word_lstm_model_with_early_stopping.h5')

# Function to predict next word
def predict_next_word(text, num_words=1):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([text])
        sequence = np.array(sequence)
        pred = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(pred, axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                text += ' ' + word
                break
    return text

st.title("Hamlet Next-Word Predictor")
st.write("Enter a phrase from Hamlet or any English text, and the model will predict the next word.")

user_input = st.text_input("Enter your text:")
num_words = st.slider("Number of words to predict:", 1, 5, 1)

if st.button("Predict"):
    if user_input.strip():
        result = predict_next_word(user_input, num_words)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")
