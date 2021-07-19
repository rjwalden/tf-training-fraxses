import os

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', '')
MODEL_STORAGE = os.environ.get('MODEL_STORAGE', '/var/fraxses/shared/data/models')
HIVE_ADDR = os.environ.get('HIVE_ADDR',  'hive://fraxses-csv-tft-headless:10000/default')
VERBOSE = os.environ.get('VERBOSE', True)