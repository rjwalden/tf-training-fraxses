import os
import ray
from flask import Flask, request
app = Flask(__name__)

import logging
logging.basicConfig(level=logging.DEBUG)

from models import *
from utils.helpers import get_latest_version, ray_connect

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
            config = {'organization': organization, 'dataset': dataset, 'index': request.args.get('index')}
            ray_connect()
            model_class = eval(model_type)
            model = model_class.remote(config)
        except Exception as e:
            logging.debug('failed to init')
            return str(e) 
        else:
            model.predict.remote()
            df_obj = model.get_df.remote()
            df = ray.get(df_obj)
            return df.to_json(orient='records')
    return  f'{organization}/{dataset}/{model_type} does not exist, ' + \
        'please use our training api to create one.'

if __name__ == '__main__':
   app.run(debug = True)
   # curl 'localhost:5000/prediction?model_type=AnomalyDetector&dataset=odc_ap_demo_020&organization=rosenblatt'
   # curl 'localhost:5000/prediction?model_type=AnomalyDetector&dataset=odc_ap_demo_020&organization=rosenblatt&index=0'