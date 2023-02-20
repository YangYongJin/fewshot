"""
This code allows you to train a vanilla multi-domain learning network.
"""

import os
import sys
import torch
import numpy as np
import tensorflow as tf
from time import sleep
import matplotlib.pyplot as plt
import torch.nn as nn

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator
from config import args, BATCHSIZES, LOSSWEIGHTS
from torchvision.models import resnet101
from scipy.stats import wasserstein_distance
tf.compat.v1.disable_eager_execution()
from pyemd import emd
import cv2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def train():
    # initialize datasets and loaders
    print(args['data.train'])
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    train_loaders = []
    num_train_classes = dict()
    num_features = dict()
    num_classes = dict()
    for t_indx, trainset in enumerate(trainsets):
        train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets,
                                          batch_size=32))
        num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        num_features[trainset] = []
        num_classes[trainset] = []
        for i in range(num_train_classes[trainset]):
            num_features[trainset].append([])
            num_classes[trainset].append(0)
    # testsets = MetaDatasetBatchReader('test', [trainsets], valsets, testsets, batch_size=1)

    model = resnet101(pretrained=True)
    model.fc = Identity()
    model = model.cuda()

    # Training loop
    max_iter = 20000
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    trucks = []
    with tf.compat.v1.Session(config=config) as session:
        for t_indx, train_loader in enumerate(train_loaders):
            for i in tqdm(range(max_iter)):
                
                sample = train_loader.get_train_batch(session)
                with torch.no_grad():
                    features = model(sample['images'].cuda())
                labels = sample['labels'].view(-1,1)
                for feature, label in zip(features, labels):
                    num_features[trainsets[t_indx]][int(label)].append(feature)
                    num_classes[trainsets[t_indx]][int(label)] += 1


    for t_indx, trainset in enumerate(trainsets):
        for label in range(num_train_classes[trainsets[t_indx]]):
            if num_classes[trainset][int(label)] > 0:
                num_features[trainset][int(label)] = torch.stack(num_features[trainset][int(label)]).squeeze(1).mean(0)                
        num_classes[trainset] = [ele for ele in num_classes[trainset] if ele != 0]
        num_features[trainset] = [ele for ele in num_features[trainset] if len(ele) != 0]
        num_classes[trainset] = np.asarray(num_classes[trainset], dtype=np.float32)
        num_classes[trainset] = num_classes[trainset]/num_classes[trainset].sum()

    base_dataset = 'ilsvrc_2012'
    base_features  = num_features[base_dataset]
    base_classes  = num_classes[base_dataset]

    for t_indx, trainset in enumerate(trainsets):
        print(trainset)
        if trainset!= base_dataset:
            with torch.no_grad():
                dist = torch.sqrt(torch.pow(torch.stack(base_features).unsqueeze(1).to('cpu') - torch.stack(num_features[trainset]).unsqueeze(0).to('cpu'), 2).sum(-1)).to('cpu').numpy().astype(np.float32)

            base_hist = base_classes
            compare_hist = num_classes[trainset]
            
            d,_,_ = cv2.EMD(base_hist, compare_hist, distType=cv2.DIST_USER,cost=dist)
            sim = np.exp(-0.1*d)
            print(sim)


if __name__ == '__main__':
    train()
