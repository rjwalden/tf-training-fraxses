# Standard
from __future__ import print_function
import os
import json
import logging
from python_fraxses_wrapper.wrapper import FraxsesWrapper
from dataclasses import dataclass
from python_fraxses_wrapper.error import WrapperError
from python_fraxses_wrapper.response import Response, ResponseResult
from local_settings import * 
import requests

# TensorFlow
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from google.protobuf.json_format import MessageToJson

# dev
from thrift_connect import data_object_query

logging.basicConfig(level=logging.DEBUG)

TOPIC_NAME = "tensorflow_train_request"

@dataclass
class PredictionParameters:
    data: str

@dataclass
class FraxsesPayload:
    id: str
    obfuscate: bool
    payload: PredictionParameters

def handle_message(message):
    try:
        data = message.payload
        try:
            data = data.payload['data']
        except Exception as e:
            return "Error in payload parsing 'data' element, " + str(e) 
        try:
            model = data['model']
            training_data_object = data['data_object']
            df = data_object_query(training_data_object)
            print(df.head(5))
            # DO SOME TRAINING AND SAVE MODEL HERE
            # SOME CHANGES TO MY FILE
            return df.to_dict(orient="records")
        except Exception as e:
            return "Error in payload parsing 'data_object' element, " + str(e)
    except Exception as e:
        print("Error in wrapper parsing", str(e))
        return str(e)
    return prediction

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
                print("Successfully responded to coordinator")
            else:
                error = message
                print("Error in wrapper", error.format_error(), message)
                error.log()


