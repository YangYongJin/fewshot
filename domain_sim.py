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
tf.compat.v1.disable_eager_execution()

def train():
    # initialize datasets and loaders
    print(args['data.train'])
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    train_loaders = []
    num_train_classes = dict()
    for t_indx, trainset in enumerate(trainsets):
        train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets,
                                          batch_size=1))
        num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
    # testsets = MetaDatasetBatchReader('test', [trainsets], valsets, testsets, batch_size=1)

    

    # Training loop
    max_iter = 20000
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    trucks = []
    with tf.compat.v1.Session(config=config) as session:
        for i in tqdm(range(max_iter)):
            for t_indx, train_loader in enumerate(train_loaders):
                sample = train_loader.get_train_batch(session)
                if sample['labels'] == 482:
                    print('fine')
                    trucks.append(sample['images'].permute(0,2,3,1).squeeze(0).to('cpu'))
            if (len(trucks) >= 4):
                break

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')
    ax1.imshow(trucks[0])
    ax2.imshow(trucks[1])
    ax3.imshow(trucks[2])
    ax4.imshow(trucks[3])

    plt.savefig('books_read.png')

    


if __name__ == '__main__':
    train()
