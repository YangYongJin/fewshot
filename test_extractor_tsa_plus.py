"""
This code allows you to evaluate performance of a single feature extractor + tsa with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

To test the url model with residual adapters in matrix form and pre-classifier alignment
on the test splits of all datasets, run:
python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url -test.tsa-ad-type residual \
--test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye

To test the url model with residual adapters in matrix form and pre-classifier alignment
on the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url -test.tsa-ad-type residual \
--test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye \
-data.test ilsrvc_2012 dtd vgg_flower quickdraw
"""

import os
import torch
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #added part
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.tsa_plus import resnet_tsa_plus, tsa_plus
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def evaluate_t_sne(features, labels):
        actual = []
        deep_features = []

        with torch.no_grad():

            deep_features = features.cpu().numpy().tolist()
            actual = labels.cpu().numpy().tolist()

        tsne = TSNE(n_components=2, random_state=0) # 사실 easy 함 sklearn 사용하니..
        cluster = np.array(tsne.fit_transform(np.array(deep_features)))
        actual = np.array(actual)

        plt.figure(figsize=(10, 10))
        n_way = len(labels.unique())
        for i in range(n_way):
            idx = np.where(actual == i)
            plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=str(i))

        plt.legend()
        plt.savefig('books_read.png')
        # 1/0


def main():
    TEST_SIZE = args['test.size']

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    if args['data.test'] == 'All':
        testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    if args['test.mode'] == 'mdl':
        # multi-domain learning setting, meta-train on 8 training sets
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        # single-domain learning setting, meta-train on ImageNet
        trainsets = ['ilsvrc_2012']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    model = resnet_tsa_plus(model)
    model.reset()
    model.cuda()

    accs_names = ['NCC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            if dataset in trainsets:
                scale = 0.1
            else:
                scale = 1.0
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                # initialize task-specific adapters and pre-classifier alignment for each task
                model.reset()
                # loading a task containing a support set and a query set
                sample = test_loader.get_test_task(session, dataset)
                context_images = sample['context_images']
                context_labels = sample['context_labels']
        

                # optimize task-specific adapters and/or pre-classifier alignment
                used_lr, used_lr_beta, used_lrw,eff, c_features = tsa_plus(context_images, context_labels, model, max_iter=40, scale=scale, distance=args['test.distance'])
                # target_images = sample['target_images']
                target_labels = sample['target_labels']
                with torch.no_grad():
                    context_features = model.embed_concat(sample['context_images'], c_features)
                    target_features = model.embed_concat(sample['target_images'])
                    # evaluate_t_sne(target_features,target_labels)
                    # if 'beta' in args['test.tsa_opt']:
                    #     context_features = model.embed_concat(context_features)
                    #     target_features = model.embed_concat(target_features)
                _, stats_dict, _ = prototype_loss(
                    context_features, context_labels,
                    target_features, target_labels, distance=args['test.distance'])

                var_accs[dataset]['NCC'].append(stats_dict['acc'])
                if ((i+1)%5 == 0):
                    d_acc = np.array(var_accs[dataset]['NCC']) * 100
                    print(f"{dataset}: current (lr: {used_lr}, lr_beta: {used_lr_beta}, lrw: {used_lrw})(len: ({len(d_acc)})), eff: {eff} test_acc {d_acc.mean():.2f}%")
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
    # Print nice results table
    print('results of {} with {}'.format(args['model.name'], args['test.tsa_opt']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], args['out.method'])
    out_path = check_dir(out_path, True)
    out_path_txt = os.path.join(out_path, '{}-tsa-{}-{}-{}-{}-test-results-.txt'.format(args['model.name'], args['test.tsa_opt'], args['test.tsa_ad_type'], args['test.tsa_ad_form'], args['test.tsa_init']))
    out_path = os.path.join(out_path, '{}-tsa-{}-{}-{}-{}-test-results-.npy'.format(args['model.name'], args['test.tsa_opt'], args['test.tsa_ad_type'], args['test.tsa_ad_form'], args['test.tsa_init']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")
    with open(out_path_txt, 'w') as f:
        f.write(table)


if __name__ == '__main__':
    main()



