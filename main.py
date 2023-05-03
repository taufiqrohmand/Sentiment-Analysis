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


app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
    'title': LazyString(lambda:'Challenge Platinum API Documentation and Modelling Machine Learning'),
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
def hello():
    return """<h1>Home Page API for Sentiment Prediction based on Machine Learning Nural Network</h1>
    <p>Silakan untuk masuk ke fitur API bisa dengan  
    <a href = 'http://127.0.0.1:5000/docs'>klik disini</a> </p>"""

#API Text Processing Neural Network
@swag_from("docs/text_nn.yml", methods=['POST'])
@app.route('/text-Processing', methods=['POST'])
def text_processing():
    textinput = request.form.get('text')
    textoutput = clean(textinput)
     
    json_respon = {
        'input' : textinput,
        'output' : textoutput,
    }
    response_data = jsonify(json_respon)
    return response_data


if __name__ == '__main__' :
    app.run(debug=True)