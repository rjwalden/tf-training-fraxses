import re
import os
import ray
import logging
logging.basicConfig(level=logging.DEBUG)

from utils.local_settings import MODEL_STORAGE

def get_latest_version(model_type:str, organization:str, dataset:str) -> int:
    '''
    Checks file system for the latest version of a model

    Returns:

    latest_version (int): the latest version of the model

    '''    
    latest_version = 0
    model_path = os.path.join(MODEL_STORAGE, organization, dataset)
    logging.debug(f'searching filesystem for latest version of {model_path}/{model_type}')
    try:
        for f in os.listdir(model_path): 
            if re.match(rf'{model_type}_[0-9]+.pb', f):
                try:
                    v = int(f[f.index('_')+1:f.index('.pb')])
                except SyntaxError:
                    logging.warning(f'Bad file: "{f}" please remove.')
                else:
                    if v > latest_version:
                        latest_version = v
    except:
        logging.debug(f'{model_path}/{model_type} has not been trained before')
    logging.debug(f'{model_path}/{model_type} latest version: {latest_version}')
    return latest_version


def ray_connect(name:str = 'automl-cluster-ray-head', port:str = 10001):
    try:
        ray.client(address=f'{name}:{port}').connect()
        logging.debug(f'Connected to remote Ray cluster @{name}:{port}')
    except Exception as e:
        logging.error(f'Failed to connect to remote Ray cluster @{name}:{port}. ' + str(e))
        logging.debug('Running Ray locally.')
        ray.init(local_mode=True)