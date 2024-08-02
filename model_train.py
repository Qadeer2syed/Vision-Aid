#NOTE:
#This is the seconf part of our project where we train our model based on features extracted
#and labled training captions
import os
import json
from sklearn.model_selection import train_test_split
import random
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import pickle
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
from keras.utils import pad_sequences, to_categorical
import joblib

#We define 2 lists
#1st one is to store the caption IDs
#Second one is to store caption ID with corresponding captions
caption_id_list = []
data_corpus = []

#****************************************************************************
#*For fast execution we have written text but this can be easily rewritten as*
#*data_path = os.path.join('Train_data','Train_lbl.json')                 *
#*****************************************************************************

data_path = os.path.join('Testing','Testing_label.json')
with open(data_path) as file_open:
    caption_data = json.load(file_open)
#from the json file we obtain the captions and corresponding IDs
for json_obj in caption_data:
    caption_set = json_obj['caption']
    #we select 5 random captions for each ID
    caption_subset = random.choices(caption_set,k=5)
    for caption in caption_subset:
        #Encoding the start and end token to mark start and end of the caption
        caption = "<sotk> "+caption+" eotk"
        data_corpus.append([json_obj['id'],caption])
    caption_id_list.append(json_obj['id'])

#Splitting the training data into training and validation data respectively
training_list, validation_list = train_test_split(data_corpus,test_size=0.1,random_state=42)
# all captions are stored in list to tokenize them
all_captions = []

for x in data_corpus:
    all_captions.append(x[1])

#This part of code tokenizes the text and creates a dictionary representing words as key value pairs
word_tokenizer = Tokenizer(num_words=1500)
word_tokenizer.fit_on_texts(all_captions)

# all_caption_sequences = word_tokenizer.texts_to_sequences(all_captions)
# all_caption_sequences = np.array(all_caption_sequences)
# all_caption_sequences = pad_sequences(all_caption_sequences, padding='post', truncating='post',
#                                         maxlen=10)

with open('feature_dict2.pickle', 'rb') as file:
    feature_dict = pickle.load(file)

#Loading and printing our feature dictionary from the pickle file
print(feature_dict)

print("Loaded dictionary:", feature_dict)

#This function returns data formatted as encoder input, decoder input and decoder target 
def get_data(data_list):
    encoder = []
    decoder_in= []
    decoder_targ = []
    video_name = []
    video_caption = []
    for idx, cap in enumerate(data_list):
        caption = cap[1]
        video_name.append(cap[0])
        video_caption.append(caption)
    #We take the caption of each sequence and get their encodings using tokenizer
    train_sequences = word_tokenizer.texts_to_sequences(video_caption)
    train_sequences = np.array(train_sequences)
    train_sequences = pad_sequences(train_sequences, padding='post', truncating='post',
                                    maxlen=10)
    file_size = len(train_sequences)
    #we iterate through all the examples to create a training data unit for each example
    for idx in range(0, file_size):
        if video_name[idx] not in feature_dict:
            continue
        encoder.append(feature_dict[video_name[idx]])
        y = to_categorical(train_sequences[idx], 1500)
        decoder_in.append(y[:-1])
        decoder_targ.append(y[1:])

    #Finally all the training examples are stored in a numpy array
    encoder_input = np.array(encoder)
    decoder_input = np.array(decoder_in)
    decoder_target = np.array(decoder_targ)

    return ([encoder_input, decoder_input], decoder_target)


#Following is the design of our encoder-decoder model which is a standard model used throughout
#Encoder has input size of 80x4096 from our feature dictionary
#Our model has latent dimension of 512 for embeddings
encoder_inputs = Input(shape=(80, 4096), name="encoder_inputs")
encoder = LSTM(512, return_state=True, return_sequences=True, name='encoder_lstm')
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, 1500), name="decoder_inputs")
decoder_lstm = LSTM(512, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(1500, activation='relu', name='decoder_relu')
decoder_outputs = decoder_dense(decoder_outputs)

#Following shows the model structure for training
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#Our model is trained for accuracy with loss function as cross entropy
model.compile(metrics=['accuracy'], loss='categorical_crossentropy')

#we get formatted data from the function defined above
train_list = get_data(training_list)
validation_list = get_data(validation_list)

#We fit the model setting hyper parametes for early convergence and low complexity
model.fit(train_list[0], train_list[1], validation_data=(validation_list[0], validation_list[1]), validation_steps=1,
             epochs=30, steps_per_epoch=5, batch_size =1)

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

#We save our encoder and decoder models as h5 files and tokenizer for inference later
encoder_model.save(os.path.join('encoder_model.h5'))
decoder_model.save_weights(os.path.join('decoder_model_weights.h5'))
with open(os.path.join('tokenizer'), 'wb') as file:
    joblib.dump(word_tokenizer, file)

