import os
from flask import Flask, request
app = Flask(__name__)

from models import *
from utils.helpers import get_latest_version

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    try:
        model_type = request.args['model_type']
        organization = request.args['organization']
        dataset = request.args['dataset']
    except KeyError:
        return 'Invalid request, must contain all parameters'

    if get_latest_version(model_type, organization, dataset):
        try:
            model = eval(model_type)(organization, dataset)
        except Exception as e:
            return str(e)
        else:
            model.predict()
            return model.df.to_json(orient='records')
    return  f'{organization}/{dataset}/{model_type} does not exist, ' + \
        'please use our training api to create one.'

if __name__ == '__main__':
   app.run(debug = True)
   # curl 'localhost:5000/prediction?model_type=AnomalyDetector&dataset=odc_ap_demo_020&organization=rosenblatt'