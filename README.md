# Project Name: Hamlet Next-Word Predictor

## Overview
This project implements a Next-Word Prediction model using LSTM (Long Short-Term Memory) Recurrent Neural Networks, trained on the text of Shakespeare's "Hamlet." The model predicts the next word in a sequence, making it useful for text generation, creative writing assistance, and natural language processing research.

## Features
- LSTM-based neural network for next-word prediction
- Trained on the full text of "Hamlet"
- Early stopping to prevent overfitting
- Pre-trained model weights included
- Jupyter notebook for experiments and analysis
- Easy-to-use Python script for inference

## Files
- `app.py`: Main script for running the next-word prediction model
- `experiemnts.ipynb`: Jupyter notebook for model training, evaluation, and experimentation
- `hamlet.txt`: Source text used for training the model
- `next_word_lstm_model_with_early_stopping.h5`: Pre-trained model with early stopping
- `next_word_lstm.h5`: Pre-trained model without early stopping
- `requirements.txt`: List of required Python packages

## Getting Started
### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone this repository or download the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
#### Run the Prediction Script
```bash
python app.py
```

#### Experiment in Jupyter Notebook
Open `experiemnts.ipynb` in Jupyter and follow the cells to train, evaluate, or use the model interactively.

## Model Details
- **Architecture:** LSTM-based RNN
- **Dataset:** Full text of "Hamlet" by William Shakespeare
- **Output:** Predicts the next word given a sequence of words

## Applications
- Text generation
- Creative writing tools
- NLP research and education

## License
This project is for educational and research purposes. The text of "Hamlet" is in the public domain.

## Acknowledgements
- William Shakespeare for the source text
- Keras/TensorFlow for deep learning frameworks

---
**Project Name Suggestion:** Hamlet Next-Word Predictor
