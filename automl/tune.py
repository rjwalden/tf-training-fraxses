from python_fraxses_wrapper.wrapper import FraxsesWrapper
from dataclasses import dataclass
from python_fraxses_wrapper.error import WrapperError
from python_fraxses_wrapper.response import Response, ResponseResult
from utils.local_settings import * 
from utils.helpers import ray_connect

# import pyspark
import ray
# from ray.util.sgd.tf.tf_trainer import TFTrainer, TFTrainable
# from raydp.spark import RayMLDataset

import logging
logging.basicConfig(level=logging.DEBUG)

from models import *

# import databricks.koalas as ks
# ks.set_option("compute.default_index_type", "distributed")  # Use default index prevent overhead.

from utils.helpers import get_latest_version

TOPIC_NAME = 'tensorflow_train_request'

@dataclass
class PredictionParameters:
    data: str

@dataclass
class FraxsesPayload:
    id: str
    obfuscate: bool
    payload: PredictionParameters
    
def tune_model(**config):
    '''Model tuning dynamic dispatchor'''
    try:
        model_class = eval(config['model_type'])
    except:
        raise Exception(f'{config["model_type"]} is not a valid model_type.')

    ray_connect()

    if get_latest_version(config['model_type'], config['organization'], config['dataset']):
        logging.debug('fine tuning model')
        m = model_class.remote(config)
        # error = m.train.remote()
        # analysis = ray.get(error)
        # m.cleanup.remote()
        error = 0
    else:
        logging.debug('training from scratch')
        # TODO: use model_class._tune_config withn hyperopt?
        # config.update(hypeparameters)
        m = model_class.remote(config)
        error = m.train.remote()
        error = ray.get(error)
        m.cleanup.remote()
                
    logging.debug('completed tune')
    return error

def handle_message(message):
    try:
        data = message.payload
    except Exception as e:
        logging.debug("Error in wrapper parsing", str(e))
        return str(e)
    
    try:
        data = data.payload
        logging.debug(data)
    except Exception as e:
        return "Error in payload parsing 'data' element, " + str(e) 

    try:
        model_type = data['model_type']
        organization = data['organization']
        dataset = data['dataset']
    except KeyError as e:
        return "Error in payload parsing 'data_object' element, " + str(e)
    
    try:
        error = tune_model(**data) 
        logging.debug(f'model error: {error}')
    except Exception as e:
        return 'Error tuning the model, ' + str(e)
    else:
        # TODO: return results from analysis
        return f'{organization}/{dataset}/{model_type} training completed. You can now access it via our prediction API:\n\n' \
            +  f'http://localhost:8000/prediction?model_type={model_type}&organization={organization}&dataset={dataset}\n\n'

if __name__ == "__main__":
    wrapper = FraxsesWrapper(group_id="test", topic=TOPIC_NAME) 

    with wrapper as w:
        for message in wrapper.receive(FraxsesPayload):
            if type(message) is not WrapperError:
                task = handle_message(message)
                response = Response(
                    result=ResponseResult(success=True, error=""),
                    payload=task,
                )
                message.respond(response)
                ray.shutdown()
                logging.debug('Disconnected from Ray')
                logging.debug("Successfully responded to coordinator")
            else:
                error = message
                logging.debug("Error in wrapper", error.format_error(), message)
                error.log()
