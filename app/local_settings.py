import os

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', '')
