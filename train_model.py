import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from preprocessing import clean_text, prepare_dataset, derive_sentiment

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("Reviews.csv")
    
    # Take a sample for training to keep it lightweight
    df = df.sample(n=20000, random_state=42)
    
    # Prepare labels and combined text
    df = prepare_dataset(df)
    
    print("Cleaning text...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Remove empty reviews after cleaning
    df = df[df['cleaned_review'] != ""]
    
    # Encode Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])
    y = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Tokenization
    max_words = 10000
    maxlen = 150
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_review'])
    X = tokenizer.texts_to_sequences(df['cleaned_review'])
    X = pad_sequences(X, maxlen=maxlen)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Architecture
    model = Sequential([
        Embedding(max_words, 100, input_length=maxlen),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    
    # Save artifacts
    print("Saving artifacts...")
    model.save('sentiment_model.h5')
    
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('label_encoder.pkl', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Training complete!")

if __name__ == "__main__":
    train()
