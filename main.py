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
allowed_extensions = set(['csv'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions



#Api file prediction LSTM
@swag_from('docs/file_lstm.yml', methods = ['POST'])
@app.route('/file_lstm', methods=['POST'])
def file_lstm():
    file = request.files['file']

    if file and allowed_file(file.filename):
        #Rename file with original name + date time process
        filename = secure_filename(file.filename)
        time_stamp = (datetime.now().strftime('%d-%m-%Y_%H%M%S'))
        new_filename = f'{filename.split(".")[0]}_{time_stamp}.csv'
        
        #save file input to INPUT folder on local
        save_location = os.path.join('input', new_filename)
        file.save(save_location)
        filepath = 'input/' + str(new_filename)

        #Load data file input
        data = pd.read_csv(filepath, encoding='latin-1')
        first_column_pre_process = data.iloc[:, 0]

        #empety array
        cleaned_file = []

        for text in first_column_pre_process:
            file_clean = clean(text)




if __name__ == '__main__' :
    app.run(debug=True, threaded=True, port=5000)