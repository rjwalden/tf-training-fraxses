from pyhive import hive
import pandas as pd
import requests
import json
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from sqlalchemy.orm import sessionmaker

def create_session(dataObject):
    engine = create_engine('hive://fraxses-csv-tft-headless:10000/default')
    Session = sessionmaker(bind=engine)
    dataObjectEngine = Table(dataObject, MetaData(bind=engine), autoload=True)
    session = Session()
    return session, dataObjectEngine, engine

def data_object_query(dataObject):
    session, dataObjectEngine, engine = create_session(dataObject)
    query = session.query(dataObjectEngine).statement #.all()
    #print(query.compile)
    df = pd.read_sql_query(query, engine)
    #df = pd.DataFrame(session.query(dataObjectEngine).all())
    #print(df)
    return df
				
