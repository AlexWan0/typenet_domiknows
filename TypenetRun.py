import sys
sys.path.append('DomiKnowS/')

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import os
import joblib
import pickle
import torch
import argparse

from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import SolverPOIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsLoss

from TypenetGraph import app_graph

from sensors.MLPEncoder import MLPEncoder
from sensors.TypeComparison import TypeComparison
from readers.TypenetReader import WikiReader

import config

# args
parser = argparse.ArgumentParser()
parser.add_argument('--limit', dest='limit', type=int, default=None)

args = parser.parse_args()

# load data
file_data = {}

file_data['train_bags'] = h5py.File(os.path.join("resources/MIL_data/entity_bags.hdf5"), "r")
file_data['embeddings'] = np.zeros(shape=(2196018, 300))
#file_data['embeddings'] = np.load(os.path.join(self.file, 'data/pretrained_embeddings.npz'))["embeddings"]

file_data['typenet_matrix_orig'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_transitive_closure.joblib'))

with open(os.path.join('resources/data/vocab.joblib'), "rb") as file:
    file_data['vocab_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

with open(os.path.join('resources/MIL_data/entity_dict.joblib'), "rb") as file:
    file_data['entity_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

file_data['type_dict'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_type2idx.joblib'))

with open(os.path.join('resources/MIL_data/entity_type_dict_orig.joblib'), "rb") as file:
    file_data['entity_type_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

wiki_train = WikiReader(file='resources/MIL_data/train.entities', type='file', file_data=file_data, bag_size=20, limit_size=args.limit)

print('building graph')
# get graph attributes
mention = app_graph['mention']
label = app_graph['label']

# text data sensors
mention['MentionRepresentation'] = ReaderSensor(keyword='MentionRepresentation')
mention['Context'] = ReaderSensor(keyword='Context')

# label data sensors
mention[label] = ReaderSensor(keyword='GoldTypes')

# module learners
mention['encoded'] = ModuleLearner('Context', 'MentionRepresentation', module=MLPEncoder(pretrained_embeddings=file_data['embeddings'], mention_dim=file_data['embeddings'].shape[-1]))
mention[label] = ModuleLearner('encoded', module=TypeComparison(config.num_types, config.type_embed_dim))

# create program
program = SolverPOIProgram(
    app_graph,
    loss=MacroAverageTracker(BCEWithLogitsLoss()),
    metric=PRF1Tracker(DatanodeCMMetric())
    )

print('training')
# train
program.train(wiki_train, train_epoch_num=1, Optim=torch.optim.Adam, device='cpu')

