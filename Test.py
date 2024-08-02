#NOTE:
#This is the third part of our project
#In this we try to infer captions from the testing videos using our trained saved models
import os
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import joblib
import functools
import operator
import os
import time
import numpy as np
import json
import random
import pickle
from bert_score import score as bert_score
import torch
import pyttsx3

#We load the tokenizer that we saved in the training file
with open(os.path.join('tokenizer'), 'rb') as file:
        tokenizer = joblib.load(file)

#We load the trained encoder and decoder models from previous part
#Moreover we define the structure decoder model
#Notice that the structure of the model remains same
Trained_encoder = load_model(os.path.join('encoder_model.h5'))
decoder_inputs = Input(shape=(None, 1500))
#Softmax is used here to covert the prediction of tokens into probabilities
decoder_dense = Dense(1500, activation='softmax')
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
prediction_decoder = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
prediction_decoder.load_weights(os.path.join('decoder_model_weights.h5'))

#From the testing dataset we load the captions for reference and get corresponding IDs
data_corpus = []
data_path = os.path.join('Testing','Testing_label.json')
with open(data_path) as file_open:
    caption_data = json.load(file_open)

#We load the feature dictionary
#This has to be done after extracting features from the testing data
with open('feature_dict2.pickle', 'rb') as file:
    feature_dict = pickle.load(file)

#We create lists to store the following:
#The names are self-explainatory
test_set_caption = []
test_set_name=[]
test_set_features = []
predicted_summaries = []
reference_summaries = []

#Firstly we get the testing captions and their IDs
#Captions are stored for reference after prediction and IDs are used to get features
for json_obj in caption_data:
    caption_set = json_obj['caption']
    caption = random.choice(caption_set)
    reference_summaries.append(caption)
    test_set_caption.append(caption)
    test_set_name.append(json_obj['id'])
    test_set_features.append(feature_dict[json_obj['id']])
    
#We define the reverse as we want to reverse map from numbers to words
reverse_index = {value: key for key, value in tokenizer.word_index.items()}

#Iterating over all the testing data
for idx in range(0,len(test_set_name)):
    feature_array = test_set_features[idx]
    print(feature_array.shape)
    print(feature_array.size)
    #feature_array = feature_array.reshape(-1, 80, 4096)

    # we get the states after the encoder is fed the features 
    #These states are used as initial states for our decoder
    states = Trained_encoder.predict(feature_array.reshape(-1,80,4096))
    #We start with an empty targget sequence which gets the predictions for tokens
    #We mark the starting token to already be present
    target_seq = np.zeros((1, 1, 1500))
    #We create a string to store tokens for sentence
    sentence = ''
    target_seq[0, 0, tokenizer.word_index['sotk']] = 1
    #we set the max limit of words in a sentence as 10, it can terminate early if end of token eotk is reached
    for i in range(10):
        predicted_tokens, h, c = prediction_decoder.predict([target_seq] + states)
        states = [h, c]
        output_tokens = predicted_tokens.reshape(1500)
        #We get the token that has the highest probability of occuring in our sentence
        # Note that this comes from softmax we are using to define decoder above
        reverse_map = np.argmax(predicted_tokens)
        if reverse_map == 0:
            continue
        if reverse_index[reverse_map] is None:
            break
        if reverse_index[reverse_map] == 'eotk':
            break
        else:
            sentence = sentence + reverse_index[reverse_map] + ' '
            target_seq = np.zeros((1, 1, 1500))
            target_seq[0, 0, reverse_map] = 1
    #Here we get our final sentence from our decoder, we store it in predicted summaries        
    predicted_sentence = ' '.join(sentence.split()[:-1])
    print(predicted_sentence)
    predicted_summaries.append(predicted_sentence)
    engine = pyttsx3.init()
    engine.say(predicted_sentence)
    engine.runAndWait()



#We are using BERTScore as our metric to compare between predicted and actual summaries
device = "cuda" if torch.cuda.is_available() else "cpu"
#calculating the BERTScore
bert_P, bert_R, bert_F1 = bert_score(predicted_summaries, reference_summaries, lang="en", device=device)

print("BERTScore - Precision:", bert_P.mean().item())
print("BERTScore - Recall:", bert_R.mean().item())
print("BERTScore - F1 Score:", bert_F1.mean().item())
