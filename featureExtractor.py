#NOTE 
#PHASE - 1 - This is the first part of project
#feature extractor is run to get the features from the videos
#It takes a command line argument as Train_data or Testing depending on which videos we want to train
import shutil
import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import sys
import pickle

#count = 0

#This function extracts all the frames from a given video, It takes video ID as argument
def extractFrames(video):
    #array in which we store our frames
    frames = []
    #This function gets the exact path of the video based on our argument and videoID
    videopath = os.path.join(extractorType,'videos',video)
    print(videopath)
    #CV2 function to capture visual contents of video
    capture = cv2.VideoCapture(videopath)
    while capture.isOpened():
        returnType, currframe = capture.read()
        if returnType is False:
            break
        #resizing the captured frame to make it compatible with VGG16
        currframe = cv2.resize(currframe, (224, 224))
        currframe_rgb = cv2.cvtColor(currframe, cv2.COLOR_BGR2RGB)
        currframe_rgb_expanded = np.expand_dims(currframe_rgb, axis=0)
        frames.append(currframe_rgb_expanded)
    #our CV2 object releases it hold on video and returns the array of video frames
    capture.release()
    cv2.destroyAllWindows()
    return frames

#This is the calling function for our above function to extract frames and returns
#an array containing 80 frames
def getFeatures(cnn_pretrained_features,video):
    #function call to extract features
    frames = extractFrames(video)
    #select 80 frames randomly in sequence
    frame_idx = np.random.permutation(len(frames))[:80]
    frames = np.array(frames)
    #Features from frames are stored in numpy array
    frames = frames[frame_idx]
    featureDict[video] = []
    #Iterating over each frame and extracting features from VGG16
    for frame in frames:
        features = cnn_pretrained_features.predict(frame, batch_size=128)
        featureDict[video].append(features)
    if(len(featureDict[video])==0):
        return 0
    #Convert the list in our dictionary into NumPy array
    featureDict[video] = np.vstack(featureDict[video])
    return 1
    # print(featureDict[video].shape)
    # print(featureDict[video])
        

#Intnitializing our feature dictionary
featureDict = {}
#Setting up for command line argument
extractorType = sys.argv[1]
extractorType = str(extractorType)
print(extractorType)
#defining our VGG16 model
cnn_pretrained = VGG16(weights="imagenet",include_top=True,input_shape=(224, 224, 3))
#Modifiying our model get features from last 2nd layer of VGG16 model
cnn_pretrained_features = Model(inputs=cnn_pretrained.input,outputs=cnn_pretrained.layers[-2].output)
#capturing the names of the videos
videoList = os.listdir(os.path.join(extractorType, 'videos'))
#iterating over videos to get features
for i in videoList:
    getFeatures(cnn_pretrained_features,i)

#we want to save the dictionary to use it in other programs
#It is saved as a pickle file
file_path = 'feature_dict2.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(featureDict, file)

