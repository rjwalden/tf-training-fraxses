import os
import abc
import json
import logging
import pandas as pd
import tensorflow as tf
from sqlalchemy import Table
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.orm import sessionmaker

from utils.helpers import get_latest_version
from utils.local_settings import MODEL_STORAGE, HIVE_ADDR

class FraxsesModel(object):
    __metaclass__ = abc.ABCMeta
    
    # @property
    # def _tune_config(self):
    #     raise NotImplementedError
    
    # @property
    # def bookmark(self):
    #     raise NotImplementedError

    # @property
    # def model(self):
    #     raise NotImplementedError

    # @property
    # def df(self):
    #     raise NotImplementedError

    # @_tune_config.setter
    # def _tune_config(self, value):
    #     self._tune_config = value
    
    # @bookmark.setter
    # def bookmark(self, value):
    #     self.bookmark = value

    # @model.setter
    # def model(self, value):
    #     self.model = value

    # @df.setter
    # def df(self, value):
    #     self.df = value

    def __init__(self, config:dict):
        '''Setup training configuration'''
        model_type = self.__class__.__name__
        self.logger = logging.getLogger(model_type)
        self.logger.setLevel(logging.DEBUG)

        try:
            self._organization = config['organization']
            self._dataset = config['dataset']
        except KeyError as e:
            self.logger.error(f'Invalid Configuration: {e}')

        self.version = get_latest_version(model_type, self._organization, self._dataset)
        # TODO: https://docs.ray.io/en/master/ray-design-patterns/closure-capture.html
        self.df = self.dataset_query()

        if self.version:
            config = self.load_config()
            for key in config:
                setattr(self, key, config[key])
            self.load_checkpoint()
            self.df = self.df.loc[config.get('index', self.bookmark):]

    def dataset_query(self):
        engine = create_engine(HIVE_ADDR)
        Session = sessionmaker(bind=engine)
        dataObjectEngine = Table(self._dataset, MetaData(bind=engine), autoload=True)
        session = Session()
        query = session.query(dataObjectEngine).statement
        df = pd.read_sql_query(query, engine)
        return df

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

    def save_config(self, **config) -> None:
        '''Stores configuration kwargs as a .json'''
        fname =  f"{self.__class__.__name__}_{self.version}.json"
        config_path = os.path.join(MODEL_STORAGE, self._organization, self._dataset, fname)
        self.logger.debug(f'saving config to {config_path}')
        with open(config_path, 'w') as outfile:
            json.dump(config, outfile)

    def load_checkpoint(self):
        '''Reads .pb file for model from the filesystem '''
        path = os.path.join(MODEL_STORAGE, self._organization, self._dataset, f'{self.__class__.__name__}_{self.version}.pb')
        self.model = tf.saved_model.load(path)

    def save_checkpoint(self) -> None:
        '''Stores model object as a .pb'''
        fname = f"{self.__class__.__name__}_{self.version}.pb"
        model_path = os.path.join(MODEL_STORAGE, self._organization, self._dataset, fname)
        self.logger.debug(f'saving model to {model_path}')
        tf.saved_model.save(self.model, model_path)

    @abc.abstractmethod
    def build_model(self):
        '''Builds model'''

    @abc.abstractmethod
    def train(self):
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
