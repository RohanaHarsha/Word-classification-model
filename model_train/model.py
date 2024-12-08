import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

sentences = [
    # Real Estate Sentences (Label 1)
    'This house has three bedrooms and a spacious backyard.',
    'Looking to buy a house with a modern kitchen.',
    'I need a rental with at least two bathrooms.',
    'Renting a townhouse with access to a swimming pool.',
    'Looking for a loft apartment in a trendy neighborhood.',
    'This apartment has hardwood floors and updated appliances.',
    'Looking for a house with a large backyard for gardening.',
    'Renting a furnished apartment for short-term stay.',
    'Looking to buy a waterfront property with a dock.',
    'This house is located in a quiet residential area.',
    'Looking for a duplex with separate entrances.',
    'Renting a penthouse with panoramic views of the city.',
    'This condo has stainless steel appliances.',

    'Looking for a penthouse with a rooftop terrace.',
    'This loft apartment has exposed brick walls.',
    'Renting a luxury apartment with concierge service.',
    'This villa has a private pool and stunning ocean views.',
    'Seeking a spacious family home with a large yard.',
    'Looking for a modern condo in a walkable neighborhood.',
    'This townhouse is move-in ready and comes with all appliances.',
    'Beautiful 3-bedroom house for sale in a desirable school district.',
    'Newly renovated apartment in a historic building.',
    'Looking to buy a fixer-upper with potential for investment.',
    'Renting a cozy cabin in the mountains for a weekend getaway.',
    'Looking for a farm house with plenty of acreage.',
    'Modern apartment building with rooftop terrace and gym.',
    'Studio apartment available for rent in a prime location.',
    'Looking to buy a commercial property for a small business.',
    'Renting a warehouse space for storage and distribution.',
    'This house has a large kitchen with granite countertops and stainless steel appliances.',
    'Searching for a beachfront condo with easy access to water activities.',
    'This property has a spacious backyard perfect for entertaining guests.',
    'Seeking a quiet cottage in a rural setting with scenic views.',
    'Renting a furnished apartment with all utilities included.',
    'Looking for a pet-friendly apartment with a dog park nearby.',
    'This condo has a balcony with breathtaking city views.',
    'Newly built townhouses with modern finishes and energy-efficient features.',
    'Seeking a spacious home with a home office and ample storage.',
    'This property is ideal for a growing family with multiple bedrooms and a large yard.',
    'Looking for a rental property with a flexible lease agreement.',
    'This apartment complex has a swimming pool, fitness center, and community room.',
    'Spacious loft with high ceilings and industrial-style design.',
    'Beautiful Victorian home with original woodwork and stained glass windows.',
    'Seeking a charming cottage with a white picket fence and garden.',
    'This house has a gourmet kitchen with a wine cellar and breakfast nook.',

    # Non-Real Estate Sentences (Label 0)
    'What time is the next train to Colombo?',
    'Can you recommend a good restaurant near here?',
    'What are the best hiking trails in Sri Lanka?',
    'How do I get to the nearest beach?',
    'Where can I buy fresh produce in Badulla?',
    'What are the must-see attractions in Ella?',
    'Where can I find a good tuk-tuk driver?',
    'How much does it cost to hire a car for a day?',
    'What''s the best time of year to visit Sri Lanka?',
    'Can you tell me about the local culture and traditions?',
    'Where can I learn to surf in Sri Lanka?',
    'What are some good souvenirs to buy in Sri Lanka?',
    "What's the weather like in Nuwara Eliya?",
    'How do I get a SIM card for my phone?',
    'Where can I try authentic Sri Lankan cuisine?',
    'What are the local customs and etiquette?',
    'Where can I find a good place to get a massage?',
    'What are some interesting historical sites in Sri Lanka?',
    'How do I say "thank you" in Sinhala?',
    'Where can I learn about Buddhist teachings in Sri Lanka?',
    'What are some good books about Sri Lankan history?',
    'Can you recommend a reliable tour guide?',
    'Where can I find information about local festivals and events?',
    'How do I get to Adam''s Peak from Badulla''?',
    'What are some good places to go birdwatching in Sri Lanka?',
    'Where can I buy traditional Sri Lankan handicrafts?',
    'How do I get to the tea plantations in Nuwara Eliya?',
    'What are some good places to go camping in Sri Lanka?',
    'Where can I find a good place to practice yoga?',
    'How can I learn more about the wildlife in Sri Lanka?'
]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]


# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')


labels = np.array(labels)


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=padded_sequences.shape[1]),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if padded_sequences.shape[0] != labels.shape[0]:
    min_length = min(padded_sequences.shape[0], labels.shape[0])
    padded_sequences = padded_sequences[:min_length]
    labels = labels[:min_length]
    
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels)).batch(2)


model.fit(dataset, epochs=10, verbose=1)


model.save('nlp_model.h5')
