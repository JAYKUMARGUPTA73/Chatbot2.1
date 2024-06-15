# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:46:51 2024
@author: Jay kumar gupta
"""
import sys
sys.excepthook = lambda type, value, traceback: print(f"Uncaught exception: {value}")

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random

print("Loading chatbot components...")
try:
    model = load_model('chatbot_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    with open('intent.json', 'r', encoding="utf8") as f:
        intents = json.load(f)
    print("Intents loaded successfully")
except Exception as e:
    print(f"Error loading intents: {e}")
    sys.exit(1)

try:
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    print("Pickle files loaded successfully")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    sys.exit(1)

print("All chatbot components loaded successfully.")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.4
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
    else:
        result = "I'm sorry, I didn't understand that. Could you please rephrase?"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

''' Flask code '''
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello():
    return jsonify({"key" : "home page value"})

def decrypt(msg):
    return msg.replace("+", " ")

@app.route('/home', methods=['GET'])
def hello_name():
    name = request.args.get('name', '')
    dec_msg = decrypt(name)
    response = chatbot_response(dec_msg)
    json_obj = jsonify({"top" : {"res" : response}})
    return json_obj

@app.errorhandler(Exception)
def handle_error(e):
    print(f"An error occurred: {e}")
    return jsonify(error=str(e)), 500
