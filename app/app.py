# Standard
from __future__ import print_function
import os
import re
import abc
import json
import grpc
import random

# from tensorflow.keras import callbacks

# from datetime import datetime
# import requests

from python_fraxses_wrapper.wrapper import FraxsesWrapper
from dataclasses import dataclass
from python_fraxses_wrapper.error import WrapperError
from python_fraxses_wrapper.response import Response, ResponseResult
from local_settings import * 

# DS Libraries
import tensorflow as tf
# import pyspark
import ray
from ray import tune
from ray import serve
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
# from ray.util.sgd.tf.tf_trainer import TFTrainer, TFTrainable
# from raydp.spark import RayMLDataset

import numpy as np
import pandas as pd

# import databricks.koalas as ks
# ks.set_option("compute.default_index_type", "distributed")  # Use default index prevent overhead.

from thrift_connect import data_object_spark_query

import logging
logging.basicConfig(level=logging.DEBUG)

TOPIC_NAME = "tensorflow_train_request"
MODEL_STORAGE = '/var/fraxses/shared/data/models'
EARLY_STOPPING_PATIENCE = 5
CV_SPLITS = 5 
SYNTHETIC_ANOMALY_FREQUENCY = 1/CV_SPLITS
NUM_WORKERS = 4
VERBOSE= True
NUM_REPLICAS = 1
USE_GPU = False

# @dataclass
# class ModelParameters:
#     organization: str
#     model_type: str
#     dataset: str

@dataclass
class PredictionParameters:
    data: str

@dataclass
class FraxsesPayload:
    id: str
    obfuscate: bool
    payload: PredictionParameters
    
class FraxsesModel(tune.Trainable):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(FraxsesModel, self).__init__()

    @property
    def _tune_config(self):
        raise NotImplementedError
    
    @property
    def _organization(self):
        raise NotImplementedError
    
    @property
    def _dataset(self):
        raise NotImplementedError

    @property
    def bookmark(self):
        raise NotImplementedError

    @property
    def version(self):
        raise NotImplementedError

    def load_config(self) -> dict:
        '''
        Reads .json file for the configuration from the filesystem

        Returns: 

        config (dict): the configuration used in the last version
        '''
        path = os.path.join(MODEL_STORAGE, self._organization, self._dataset, f'{self.__class__.__name__}_{self.version}.json')
        with open(path) as json_file:
            config = json.load(json_file)
        return config

    def save_config(self) -> None:
        '''Stores configuration dictionary as a .json'''
        config_path =  f"{self._organization}/{self._dataset}/{self.__class__.__name__}_{self.version}.json"
        logging.debug(f'saving config to {config_path}...')
        with open(config_path, 'w') as outfile:
            json.dump(self.config, outfile)

    def load_checkpoint(self):
        '''Reads .pb file for model from the filesystem '''
        path = os.path.join(MODEL_STORAGE, self._organization, self._dataset, f'{self.__class__.__name__}_{self.version}.pb')
        self.model = tf.saved_model.load(path)

    def save_checkpoint(self) -> None:
        '''Stores model object as a .pb'''
        model_path = f"{self._organization}/{self._dataset}/{self.__class__.__name__}_{self.version}.pb"
        logging.debug(f'saving model to {model_path}...')
        tf.saved_model.save(self.model, model_path)

    def setup(self, config):
        '''Setup training configuration'''
        model_type = self.__class__.__name__
        self._organization = config['organization']
        self._dataset = config['dataset']
        self.version = get_latest_version(model_type, self._organization, self._dataset)
        if self.version:
            config = self.load_config()
            for key in config:
                setattr(self, key, config[key])
            self.load_checkpoint()
            self.df = data_object_spark_query(self._dataset)
            self.df = self.df.loc[self.bookmark:]
            return

    @abc.abstractmethod
    def build_model(self):
        '''Builds model'''

    @abc.abstractmethod
    def step(self):
        '''Trains model on data'''

    @abc.abstractmethod
    def cleanup(self):
        '''Cleanup procedure after completing training'''

    @abc.abstractmethod
    def predict(self):
        '''Make a prediction'''

    @abc.abstractmethod
    def lr_schedule(self):
        '''Schedule for learning rate'''

class AnomalyDetector(FraxsesModel):

    _tune_config = {
        'epochs': tune.uniform(0, 20),
        'dropout': tune.uniform(0, 1),
        'batch_size': tune.choice([8,16,32,64,128]),
        'optimizer': tune.choice(['adam']),
        'learning_rate': tune.uniform(0, 0.01),
        'activation_fn': tune.choice(['relu', 'tanh']),
        'loss_fn': tune.choice(['mse']),
        'window': tune.choice([60, 7, 30, 12, 52, 24]),
    }

    def __init__(self):
        super(AnomalyDetector, self).__init__()

    def setup(self, config:dict):
        super(AnomalyDetector, self).setup()
        try:
            self._organization = config['organization']
            self._dataset = config['dataset']
            self.df = data_object_spark_query(self._dataset)
            # self.target_col = config.get('target_schema')[0][0]
            self.index_schema = config['index_schema']
            self.context_schema = config['context_schema']
            self.target_schema = config['target_schema']
        except KeyError as e:
            logging.error(f'Invalid Configuration: {e}')
        else:
            # TODO: add plasicity
            self.window = config.get('window', 10)
            self.epochs = config.get('epochs', 10)
            self.dropout = config.get('dropout', 0.2)
            self.batch_size = config.get('batch_size', 8)
            self.optimizer = config.get('optimizer', 'adam')
            self.learning_rate = config.get('learning_rate', 0.0001)
            self.activation_fn = config.get('activation_fn', 'relu')
            self.loss_fn = config.get('loss_fn', 'mse')
            self.bookmark = self.df.iloc[0].name # TODO: should be end of training set after training

    def build_model(self) -> tf.Module:
        '''Builds a Tensorflow convolution autoencoder'''
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=self.input_shape
                ),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Dropout(rate=self.dropout),
                tf.keras.layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Dropout(rate=self.dropout),
                tf.keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Conv1DTranspose(
                    filters=1, kernel_size=7, padding="same"
                ),
            ]
        )

        return model

    def format_data_train(self) -> tuple:
        '''
        Format the dataframe for time series anomaly detection

        Args:

        df (pd.DataFrame): the dataframe to format for trai

        Returns:

        train_dataset (np.ndarray):
        test_dataset (np.ndarray):
        '''
        df = self.df # .to_koalas()   
        # mtype = [numeric, categorical, days, months, weeks, hours, minutes, seconds]
        
        index_col_nm, index_col_dtype, index_col_mtype = zip(*self.index_schema)
        context_col_nm, context_col_dtype, context_col_mtype = zip(*self.context_schema)
        target_col_nm, target_col_dtype, target_col_mtype = zip(*self.target_schema)

        cols = context_col_nm + target_col_nm + index_col_nm
        dtypes = context_col_dtype + target_col_dtype + index_col_dtype
        mtypes = context_col_mtype + target_col_mtype + index_col_mtype

        # drop unused columns
        df = df.drop([c for c in df.columns if c not in cols])

        # enforce schema
        df = df.astype({k:v for k,v in zip(cols, dtypes)})
        
        # set index
        df.index = index_col_nm
        
        # engineer features
        categorical = []
        numeric = []
        
        for i,c in enumerate(cols):
            if mtypes[i] == 'numeric':
                numeric.append(c)
            elif mtypes[i] == 'categorical':
                categorical.append(c)

        # one hot encode the categorical data
        if categorical:
            df = pd.get_dummies(df, columns=categorical, prefix=categorical) 

        sequences = []
        anomaly_cols = numeric + target_col_nm
        total_sequences = len(df) - self.window + 1
        anomaly_insertion_count = int(len(df) * SYNTHETIC_ANOMALY_FREQUENCY)
        anomaly_insertion_period = int(np.floor(total_sequences / anomaly_insertion_count))
        df['synthetic_anomaly'] = 0
        anomaly_idx = 0

        # create sequences 
        for i in range(0, total_sequences):
            sequence = df.iloc[i : (i + self.window)]
            anomaly_labels = np.zeros((1, self.window))

            # add random synthetic anomaly, used std rule for magnitude      
            if i > 0 and i % anomaly_insertion_period == 0:
                # choose a random index for the anomaly
                anomaly_idx = random.randint(0, len(sequence))

                # make the anomaly an outlier value
                quartiles = random.choice([-3,3])
                z = np.ceil(sequence[anomaly_cols].std() % 10) * random.randint(-5,5)
                anomaly_value = sequence[anomaly_cols].mean() + quartiles*sequence[anomaly_cols].std() + z

                # insert the anomaly into the sequence
                sequence[anomaly_cols].iloc[anomaly_idx] += anomaly_value

                # add the current index to the anomaly index list
                self.df['synthetic_anomaly'].iloc[i] = 1
                anomaly_labels[anomaly_idx] = 1
            elif anomaly_idx:
                # adjuist the anomaly index to insert it into the next sequence if within the sequence range
                anomaly_idx -= 1

                # insert the anomaly into the sequence
                sequence[anomaly_cols].iloc[anomaly_idx] += anomaly_value
                anomaly_labels[anomaly_idx] = 1

            sequences.append(np.stack(sequence.values, anomaly_labels))

        # split data into cross-validation datasets
        train = []
        test = []
        splits = list(map(np.stack, np.array_split(sequences, CV_SPLITS)))

        for i in range(0, 3):
            tr = np.concatenate(splits[0:2+i])
            te = splits[2+i]
            
            # normalize the numeric data
            if numeric:
                numeric_idx = [df.columns.get_loc(c) for c in numeric if c in df]
                
                # normalize training data independently
                tr_mean = tr[numeric_idx].mean()
                tr_std = tr[numeric_idx].std()
                tr[numeric_idx] = (tr[numeric_idx] - tr_mean) / tr_std

                # normalize testing data independently
                te_mean = te[numeric_idx].mean()
                te_std = te[numeric_idx].std()
                te[numeric_idx] = (te[numeric_idx] - te_mean) / te_std
        
            train.append(tr)
            test.append(te) 

        return train, test

    def lr_schedule(self, epoch:int, lr:float):
        '''Schedule for the learning rate'''
        if epoch < self.epochs/2:
            return lr*tf.math.exp(-0.1)
        else:
            return lr*tf.math.exp(-0.01)

    def step(self):
        '''Trains the model using the current configuration'''
        train_splits, test_splits = self.format_data_train()
        callbacks = []

        if self.version:
            logging.debug(f'updating existing model to version {self.version + 1}...')
            # NOTE: https://arxiv.org/abs/2002.11770
            # TODO: momentum and learning rate selection
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_schedule, verbose=int(VERBOSE)))
        else:
            logging.debug(f'training new model ...')
            self.input_shape = (train_splits[1], train_splits[2])
            self.model = self.build_model()
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
        )

        # cross validate
        scores = []
        for train, test in zip(train_splits, test_splits):
            # remove anomaly labels
            tr = train[:,:-1]
            te = test[:,:-1]

            # train model
            history = self.model.fit(
                tr,
                tr, 
                batch_size=self.batch_size,
                epochs=np.ceil(self.epochs/2) if self.version else self.epochs,
                validation_data=self.validation_(te, te),
                callbacks=callbacks
            )
            
            # make predictions
            tr_p = self.model.predict(tr)
            te_p = self.model.predict(te)
            
            # calculate error
            tr_s = np.mean((tr_p - tr)**2)
            te_s = np.mean((te_p - te)**2)

            # define threshold based on training error
            threshold = np.mean(tr_s) + np.std(tr_s)*3

            # calculate p_score (average prediction error) [capturing normal signal]
            p_score = np.mean(te_s)

            # calculate c_score (average classification error) [capturing anomalies]
            p_anomalies = te_s > threshold
            c_score = np.mean(tf.keras.losses.binary_crossentropy(test[:,-1], p_anomalies))
            
            # calculate score (average of classification and prediction error)
            score = (p_score + c_score)/2
            scores.append(score)

        # report average score for all cross validations
        tune.repot(score=np.mean(scores))

    def cleanup(self):
        '''Invoked when training is finished'''
        self.version += 1
        self.bookmark = self.df.iloc[-1].name # TODO: should be ened of train set
        self.save_checkpoint()
        delattr(self, 'df')
        delattr(self, 'model')
        self.save_config()

    def format_data_predict(self, index=None):
        '''Format data to make predictions on new data'''
        if index is None:
            df = self.df.loc[self.bookmark]
        else:
            df = self.df.loc[index]
        # mtype = [numeric, categorical, days, months, weeks, hours, minutes, seconds]
        
        index_col_nm, index_col_dtype, index_col_mtype = zip(*self.index_schema)
        context_col_nm, context_col_dtype, context_col_mtype = zip(*self.context_schema)
        target_col_nm, target_col_dtype, target_col_mtype = zip(*self.target_schema)

        cols = context_col_nm + target_col_nm + index_col_nm
        dtypes = context_col_dtype + target_col_dtype + index_col_dtype
        mtypes = context_col_mtype + target_col_mtype + index_col_mtype

        # drop unused columns
        df = df.drop([c for c in df.columns if c not in cols])

        # enforce schema
        df = df.astype({k:v for k,v in zip(cols, dtypes)})
        
        # set index
        df.index = index_col_nm
        
        # engineer features
        categorical = []
        numeric = []
        
        for i,c in enumerate(cols):
            if mtypes[i] == 'numeric':
                numeric.append(c)
            elif mtypes[i] == 'categorical':
                categorical.append(c)

        # one hot encode the categorical data
        if categorical:
            df = pd.get_dummies(df, columns=categorical, prefix=categorical) 

        # normalize the numeric data
        if numeric:
            numeric_idx = [df.columns.get_loc(c) for c in numeric if c in df]
            mean = df[numeric_idx].mean()
            std = df[numeric_idx].std()
            df[numeric_idx] = (df[numeric_idx] - mean) / std
        
        
        # create sequences 
        sequences = []
        total_sequences = len(df) - self.window + 1
        
        for i in range(0, total_sequences):
            sequence = df.iloc[i : (i + self.window)]
            sequences.append(sequence.values)

        return np.array(sequences)

    def predict(self, index:int=None):
        '''Make predictions using the latest model'''
        if index is None:
            index = self.bookmark

        samples = self.format_data_predict(index)
        model = self.load_checkpoint()
        preds = model.predict(samples)
        error = np.mean((preds - samples)**2)
        # TODO: set a threshold during training
        # anomalies = error > self.threshold 
        # return anomalies

# class FraxsesModelManager:

def get_latest_version(model_type:str, organization:str, dataset:str) -> int:
    '''
    Checks file system for the latest version of a model

    Returns:

    latest_version (int): the latest version of the model

    '''
    latest_version = 0
    model_path = os.path.join(model_type, organization, dataset)
    for f in os.listdir(MODEL_STORAGE): 
        if re.match(rf'{model_path}_v[0-9]+.pb', f):
            try:
                v = int(f[f.rindex('v')+1])
            except SyntaxError:
                logging.warning(f'Bad file: "{f}" please remove.')
            else:
                if v > latest_version:
                    latest_version = v
    logging.debug(f'{organization}/{dataset}/{model_type} latest version: {latest_version}')
    return latest_version

def tune_model(model_type:str, organization:str, dataset:str, **config):
    '''Model tuning dynamic dispatchor'''
    try:
        model_class = eval(model_type)
    except:
        raise Exception(f'{model_type} is not a valid model_type.')
    
    try:
        port = 10001
        name = 'tensorflow-train-cluster-ray-head'
        ray.client(address=f'{name}:{port}').connect()
    except Exception as e:
        logging.error(str(e))
        raise Exception(f'Failed to connect to remote Ray cluster @{name}:{port}. ' + str(e))

    analysis = None

    if get_latest_version(model_type, organization, dataset):
        logging.debug('fine tuning model')
        # analysis = tune.run(model_class)
    else:
        logging.debug('training from scratch')
        # hyperopt_search = HyperOptSearch(
        #     metric="score",
        #     mode="min"
        # )

        # # TODO: learn this
        # asha_scheduler = ASHAScheduler(
        #     time_attr='training_iteration',
        #     metric='score',
        #     mode='min',
        #     max_t=100,
        #     grace_period=10,
        #     reduction_factor=3,
        #     brackets=1
        # )

        # config = config.update(model_class._tune_config)
        # analysis = tune.run(
        #     model_class,
        #     config=config,
        #     # search_alg=hyperopt_search,
        #     # scheduler=asha_scheduler
        # )

    return analysis
        
@serve.deployment
def predict(request):
    try:
        model_type = request.query_params['model_type']
        organization = request.query_params['organization']
        dataset = request.query_params['dataset']
    except KeyError as e:
        return e

    if get_latest_version(model_type, organization, dataset):
       model = eval(model_type)()
       model.setup()
       model.predict(index=request.query_params.get('index'))
       return model.df
    else:
        return f'{organization}/{dataset}/{model_type} does not exist, please train a model'

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
        analysis = tune_model(**data) 
    except Exception as e:
        return 'Error tuning the model, ' + str(e)
    else:
        # TODO: return results from analysis
        return f'{organization}/{dataset}/{model_type} training completed. You can now access it via our prediction API:\n\n' \
            +  f'http://localhost:8080/predict?model_type={model_type}&organization={organization}&dataset={dataset}\n\n'

if __name__ == "__main__":
    # predict.deploy()
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
                logging.debug("Successfully responded to coordinator")
            else:
                error = message
                logging.debug("Error in wrapper", error.format_error(), message)
                error.log()


