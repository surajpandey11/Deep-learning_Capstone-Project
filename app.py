from flask import Flask,jsonify,request,render_template
# from keyword_spotting_service import Keyword_Spotting_Service

# IMPORT NECESSARY LIBRARIES
import librosa 
# %matplotlib inline
# import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
# from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
# import seaborn as snsip
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tqdm import tqdm, tqdm_pandas


# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import speech_recognition as spr
# import templetes 
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_1_PATH = 'models/2nd_model.h5'


# Model saved with Keras model.save()
MODEL_2_PATH = 'models\model_2d_mfcc.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_1_PATH)
model2=tf.keras.models.load_model(MODEL_2_PATH)
# model._make_predict_function()          # Necessary



print('Model loaded. Check http://127.0.0.1:5000/')

# def prepare_data(df, n, aug, mfcc):

        # sampling_rate=44100
        # audio_duration=2.5
        
        # X = np.empty(shape=(df.shape[0], n, 216, 1))
        # input_length = sampling_rate * audio_duration
        
        # cnt = 0
        # for fname,row in df.iterrows():
        #     file_path = row[0]
        #     data, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)

        #     # Random offset / Padding
        #     if len(data) > input_length:
        #         max_offset = len(data) - input_length
        #         offset = np.random.randint(max_offset)
        #         data = data[offset:(input_length+offset)]
        #     else:
        #         if input_length > len(data):
        #             max_offset = input_length - len(data)
        #             offset = np.random.randint(max_offset)
        #         else:
        #             offset = 0
        #         data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        #     # Augmentation? 
        #     if aug == 1:
        #         data = speedNpitch(data)
            
        #     # which feature?
        #     if mfcc == 1:
                # MFCC extraction 
                # MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
                # MFCC = np.expand_dims(MFCC, axis=-1)
                # X[cnt,] = MFCC
                
            # else:
            #     # Log-melspectogram
            #     melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            #     logspec = librosa.amplitude_to_db(melspec)
            #     logspec = np.expand_dims(logspec, axis=-1)
            #     X[cnt,] = logspec
                
            # cnt += 1
        
        # return X





def model_2_prediction(file_path,model2):
    n_mfcc = 30
    sampling_rate=44100
    n=n_mfcc
    f={'file_path':[file_path]}
    df= pd.DataFrame(f)
    X = np.empty(shape=(df.shape[0],n , 216, 1))
    input_length = 44100 * 2.5
    cnt=0
    data, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)

    if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

    
    # 
    # df= pd.DataFrame(file_path, columns = ['path'])
    
    
    n_mfcc = 30
    # mfcc = prepare_data(df, n = n_mfcc, aug = 0, mfcc = 1)
    MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
    MFCC = np.expand_dims(MFCC, axis=-1)
    X[cnt,] = MFCC
    
    

    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean)/std

    prediction_model_2 = model2.predict(X)

    prediction_model_2=prediction_model_2.argmax(axis=1)
    prediction_model_2 = prediction_model_2.astype(int).flatten()
    # prediction_model_2 = (lb.inverse_transform((prediction_model_2)))

    return prediction_model_2


def model_predict(path, model):

    counter=0 

    df = pd.DataFrame(columns=['mel_spectrogram'])
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    
    #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    #temporally average spectrogram
    log_spectrogram = np.mean(db_spec, axis = 0)
    df.loc[counter] = [log_spectrogram]
    counter=counter+1 

    df_combined=pd.DataFrame(df['mel_spectrogram'].values.tolist())

    df_combined = df_combined.fillna(0)

    X_test = np.array(df_combined)

    X_test = X_test[:,:,np.newaxis]
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    # lb = LabelEncoder()

    predict = model.predict(X_test)
    predict=predict.argmax(axis=1)
    predict = predict.astype(int).flatten()
    # predict = (lb.inverse_transform((predict)))
    # predictions = pd.DataFrame({'Predicted Values': predictions})

    # predictions = (lb.inverse_transform((predict)))
    return predict

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        audio_file = request.files['file']
        # audioFile = spr.AudioFile(audio_file)
        

        # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(audio_file.filename))
        # audio_file.save(file_path)

        #  Make prediction
        preds = model_predict(audio_file, model)

        # preds2= model_2_prediction(audio_file,model2)
# def predict():
#     audio_file = request.files["file"]

    # instantiate keyword spotting service singleton and get prediction
	    # kss = Keyword_Spotting_Service()
	    # predicted_keyword = kss.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	    # os.remove(file_name)

	# # send back result as a json file
	# result = {"keyword": predicted_keyword}
	# return jsonify(result)
        # result = np.where(preds2 == np.amax(preds2))
        
        
        # return str(preds)
        return render_template('show.html', data=(preds))

#   return none
if __name__=="__main__":
    app.run(debug=True)