import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import nltk
from nltk.corpus import stopwords

# Download stop words list if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class NLPModule:
    def __init__(self):
        # Define paths for the model and tokenizer
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_train', 'nlp_model.h5')
        tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'model_train', 'tokenizer.pickle')

        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        try:
            # Load the tokenizer
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except FileNotFoundError:
            print(f"Error: Tokenizer file not found at {tokenizer_path}")
            raise
        
        try:
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            raise

    def process_query(self, query: str):
        # Convert the query to sequences and pad it
        sequence = self.tokenizer.texts_to_sequences([query.lower()])
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=30)
        return padded_sequence

    def predict(self, query: str) -> list:
        # Process the query and get predictions
        processed_query = self.process_query(query)
        prediction = self.model.predict(processed_query)
        print(f"Prediction: {prediction}")

        # Return keywords if the prediction is above the threshold
        if prediction >= 0.5:
            return self.extract_keywords(query)
        else:
            return []

    def extract_keywords(self, query: str, top_n=3) -> list:
        # Tokenize the query and filter out stop words
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stop_words and word in self.tokenizer.word_index]

        # Sort words based on their token index (more likely to be important keywords)
        sorted_keywords = sorted(filtered_words, key=lambda word: self.tokenizer.word_index[word])

        # Return the top N keywords
        top_keywords = sorted_keywords[:top_n]
        return top_keywords
