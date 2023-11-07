import os


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw_data')

TYPE_LIST = ['CollisionRecords', 'PartyRecords', 'VictimRecords']
# DTYPE = {'first_column': 'str', 'second_column': 'str'}
