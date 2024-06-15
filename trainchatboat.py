# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:29:21 2024
@author: Jay kumar gupta
"""
#import statements
import nltk
nltk.download('punkt')          
nltk.download('wordnet')       
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
#core model libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import random

#Initailisation list
words = []
classes = []
documents = []
ignore_words = ['?','!','@','$']

#use json
data_file = open('intent.json', encoding="utf8").read()
intents = json.loads(data_file)

#Populating the list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#training the bot
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)




# Convert training data to TensorFlow tensors
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)

# Define the model using TensorFlow's low-level API
def create_model():
    inputs = tf.keras.Input(shape=(len(train_x[0]),))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(train_y[0]), activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_model()

# Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(5)

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Training loop
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

epochs = 200
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch+1}")
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        if step % 200 == 0:
            print(f"Training loss (for one batch) at step {step}: {float(loss_value):.4f}")

# Save the model
model.save('chatbot_trained.h5')
print("model_created_successfully")