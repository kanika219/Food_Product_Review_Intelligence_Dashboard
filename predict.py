import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocessing import clean_text

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path, label_encoder_path):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            
        with open(label_encoder_path, 'rb') as handle:
            self.label_encoder = pickle.load(handle)
            
        self.maxlen = 150
        
    def predict(self, texts):
        if not texts:
            return [], [], []
            
        # Preprocess texts
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)
        
        # Predictions
        predictions = self.model.predict(padded_sequences, verbose=0)
        
        # Get class labels and confidence scores
        class_indices = np.argmax(predictions, axis=1)
        sentiments = self.label_encoder.inverse_transform(class_indices)
        confidence_scores = np.max(predictions, axis=1)
        
        return sentiments, confidence_scores, cleaned_texts
