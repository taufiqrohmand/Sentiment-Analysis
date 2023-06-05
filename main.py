#import library
import pandas as pd
import numpy as np
import os
import json
import pickle

from flask import Flask, jsonify, send_from_directory, make_response, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from datetime import datetime
from fileinput import filename
from werkzeug.utils import secure_filename

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model

from Cleansing import clean

#Flask
app = Flask(__name__, static_folder='docs', static_url_path='')

app.json_encoder = LazyJSONEncoder

#Swagger
swagger_template = dict(
    info = {
    'title': LazyString(lambda:'Sentiment Prediction Analysis'),
    'version' : LazyString(lambda: '2.0.0'),
    'description' : LazyString(lambda: 'Data Documentation API for Sentiment Prediction Machine Learning'),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app,template = swagger_template, config = swagger_config)

#Route Home Page
@app.route('/')
def home():
    return app.send_static_file('home.html')

#Load Feature NN
file = open('asset/feature/feature_nn.pickle', 'rb')
feature_nn = pickle.load(file)
file.close()

#Load Model NN
file = open('asset/model/model_nn.pickle', 'rb')
model_nn = pickle.load(file)
file.close()


#Load Tokenizer
file = open('asset/feature/tokenizer.pickle', 'rb')
load_token = pickle.load(file)
file.close()
sentiment = ['negative', 'neutral', 'positive']


#Load Feature LSTM
file = open('asset/feature/x_pad_sequences.pickle', 'rb')
feature_lstm = pickle.load(file)
file.close()


#load Model LSTM
model_lstm = load_model('asset/model/model_lstm.h5')


def sentiment_nn(textinput):
    text = feature_nn.transform([clean(textinput)])
    result = model_nn.predict(text)[0]
    return result

def sentiment_nnfile(textinput):
    text = feature_nn.transform([clean(textinput)])
    result = model_nn.predict(text)[0]
    return str (result)



#API Text Processing Neural Network
@swag_from("docs/text_nn.yml", methods=['POST'])
@app.route('/text-nn', methods=['POST'])
def text_nn():

    textinput = request.form.get('text')
    textoutput = clean(textinput)
    sentimentoutput = sentiment_nn(textinput)
     
    json_respon = {
        'status_code': 200,
        'description': "Result Sentiment Neural Network",
        'data': {
            'Input' : textinput,
            'Output text' : textoutput,
            'Result sentiment' : sentimentoutput
        },
    }
        
    response_data = jsonify(json_respon)
    return response_data


#API Text Processing Long Short Term Memory
@swag_from("docs/text_lstm.yml", methods=['POST'])
@app.route('/text-lstm', methods=['POST'])
def text_lstm():
    textinput = request.form.get('text')
    textoutput = clean(textinput)
    
    feature = load_token.texts_to_sequences(textoutput)
    feature = pad_sequences(feature, maxlen=feature_lstm.shape[1])

    pred_lstm = model_lstm.predict(feature)

    sentimentoutput = sentiment[np.argmax(pred_lstm[0])]
     
    json_respon = {
        'status_code': 200,
        'description': "Result Sentiment Neural Network",
        'data': {
            'Input' : textinput,
            'Output text' : textoutput,
            'Result sentiment' : sentimentoutput
        },
    }
    
    response_data = jsonify(json_respon)
    return response_data


#Allow Access File
allowed_extensions = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions


#Api file prediction NN
@swag_from('docs/file_nn.yml', methods = ['POST'])
@app.route('/file_nn', methods=['POST'])
def file_nn():
    file = request.files['file']
    new_df = pd.DataFrame()

    if file and allowed_file(file.filename):
        #Rename file with original name + date time process
        filename = secure_filename(file.filename)
        time_stamp = (datetime.now().strftime('%d-%m-%Y_%H%M%S'))
        new_filename = f'{filename.split(".")[0]}_{time_stamp}_PredictNN.csv'
        
        #save file input to INPUT folder on local
        save_location = os.path.join('input', new_filename)
        file.save(save_location)
        filepath = 'input/' + str(new_filename)

        #Load data file input
        data = pd.read_csv(filepath, encoding='latin-1')
        first_column_pre_process = data.iloc[:, 0]
    
        #clean and prediction
        input_clean = first_column_pre_process.apply(clean)
        sentiment_file = first_column_pre_process.apply(sentiment_nnfile)

        new_df = pd.DataFrame(
            {'Input': first_column_pre_process,
             'Output': input_clean,
             'Result': sentiment_file,
             })
        
        outputfilepath = f'output/{new_filename}'
        new_df.to_csv(outputfilepath)

    # Convert the DataFrame to a list of dictionaries
    data = new_df.to_dict(orient='records')

    json_respon = {
        'status_code': 200,
        'description': "Result File Sentiment Neural Network",
        'data': data
    }
        
    response_data = jsonify(json_respon)
    return response_data

#Api file prediction NN
@swag_from('docs/file_lstm.yml', methods = ['POST'])
@app.route('/file_lstm', methods=['POST'])
def file_lstm():
    file = request.files['file']
    new_df = pd.DataFrame()

    if file and allowed_file(file.filename):
        #Rename file with original name + date time process
        filename = secure_filename(file.filename)
        time_stamp = (datetime.now().strftime('%d-%m-%Y_%H%M%S'))
        new_filename = f'{filename.split(".")[0]}_{time_stamp}_PredictLSTM.csv'
        
        #save file input to INPUT folder on local
        save_location = os.path.join('input', new_filename)
        file.save(save_location)
        filepath = 'input/' + str(new_filename)

        #Load data file input
        data = pd.read_csv(filepath, encoding='latin-1')
        first_column_pre_process = data.iloc[:, 0]
    
        # Cleansing
        input_clean = first_column_pre_process.apply(clean)

        # Feature Extraxtion
        features = load_token.texts_to_sequences(input_clean)
        features = pad_sequences(features, maxlen=feature_lstm.shape[1])

        # Predict
        pred_lstm = model_lstm.predict(features)
        sentiment_file = [sentiment[np.argmax(pred)] for pred in pred_lstm]

        new_df = pd.DataFrame(
            {'Input': first_column_pre_process,
             'Output': input_clean,
             'Result': sentiment_file,
             })
        
        outputfilepath = f'output/{new_filename}'
        new_df.to_csv(outputfilepath)

    # Convert the new_df DataFrame to a list of dictionaries
    data = new_df.to_dict(orient='records')

    json_respon = {
        'status_code': 200,
        'description': "Result File Sentiment LSTM",
        'data': data
    }
        
    response_data = jsonify(json_respon)
    return response_data

# Error Handling
@app.errorhandler(400)
def handle_400_error(_error):
    "Return a http 400 error to client"
    return make_response(jsonify({'error': 'Misunderstood'}), 400)


@app.errorhandler(401)
def handle_401_error(_error):
    "Return a http 401 error to client"
    return make_response(jsonify({'error': 'Unauthorised'}), 401)


@app.errorhandler(404)
def handle_404_error(_error):
    "Return a http 404 error to client"
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(500)
def handle_500_error(_error):
    "Return a http 500 error to client"
    return make_response(jsonify({'error': 'Server error'}), 500)



if __name__ == '__main__' :
    app.run(debug=True, threaded=True, port=5000)