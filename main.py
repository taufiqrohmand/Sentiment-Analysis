#import library
import pandas as pd
import os

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
    'title': LazyString(lambda:'Sentiment Prediction Analysis for Tweet'),
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



#API Text Processing Neural Network
@swag_from("docs/text_nn.yml", methods=['POST'])
@app.route('/text-nn', methods=['POST'])
def text_nn():
    textinput = request.form.get('text')
    textoutput = clean(textinput)
    sentimentoutput = clean(textoutput)
     
    json_respon = {
        'input' : textinput,
        'output text' : textoutput,
        'output sentiment' : sentimentoutput,
    }
    response_data = jsonify(json_respon)
    return response_data


#API Text Processing Long Short Term Memory
@swag_from("docs/text_lstm.yml", methods=['POST'])
@app.route('/text-lstm', methods=['POST'])
def text_lstm():
    textinput = request.form.get('text')
    textoutput = clean(textinput)
    sentimentoutput = clean(textoutput)
     
    json_respon = {
        'input' : textinput,
        'output text' : textoutput,
        'output sentiment' : sentimentoutput,
    }
    response_data = jsonify(json_respon)
    return response_data


#Allow Access File
allow_extension = set(['csv'])
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions



#Api file prediction NN
@swag_from('docs/file_nn.yml', methods = ['POST'])
@app.route('/file_nn', methods = ['POST'])



if __name__ == '__main__' :
    app.run(debug=True, threaded=True, port=5000)