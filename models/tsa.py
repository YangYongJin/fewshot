'''
tsa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach task-specific parameters, including adapters, pre-classifier alignment (PA) mapping
from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf), to a pretrained backbone. 
It only learns attached task-specific parameters from scratch on the support set to adapt 
the pretrained model for previously unseen task with very few labeled samples.
'Cross-domain Few-shot Learning with Task-specific Adapters.' (https://arxiv.org/pdf/2107.00358.pdf)
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss, distillation_loss
from utils import device

class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            self.alpha.requires_grad = True


    def forward(self, x):
        y = self.conv(x)
        if 'alpha' in args['test.tsa_opt']:
            x = F.conv2d(x, (self.alpha), stride=self.conv.stride)

            y = (y + x) #+ self.alpha_bias #*self.alpha_weight

        return y

class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.beta_weight = nn.Parameter(torch.ones(1,feat_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        # mean = x.mean(dim=0).view(1,-1)
        # std = torch.sqrt(x.var(dim=0).view(1,-1)+1e-5)
        # x = (x-mean)*self.beta_weight/(std)
        return x

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_tsa, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
            v.requires_grad=False
        self.orig_resnet = copy.deepcopy(orig_resnet)

        # for name, m in orig_resnet.named_children():
        #     if isinstance(m, nn.Conv2d):
        #         new_conv = conv_tsa(m)
        #         setattr(orig_resnet, name, new_conv)

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k and v.requires_grad:
                # nn.init.constant_(v, 0.0)
                if 'bias' not in k:
                    if 'weight' in k:
                        v.data = torch.ones(v.size()).to(v.device)
                    else:
                        v.data=torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)*0.0001
                        # v.data = torch.rand(v.size()).to(v.device)*0.001#.unsqueeze(-1).unsqueeze(-1).to(v.device)*0.001
                    # nn.init.kaiming_normal_(v)
                    # if 'weight' in k:
                    #     v.data = torch.ones(v.size()).to(v.device)
                else:
                    nn.init.constant_(v, 0.0)

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
        # v = self.beta.beta_weight
        # v.data = torch.ones(v.size()).to(v.device)


def tsa(context_images, context_labels, model, max_iter=40, lr=0.1, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """

    model.eval()
    tsa_opt = args['test.tsa_opt']
    alpha_params = [v for k, v in model.named_parameters() if ('alpha' in k and 'alpha_weight' not in k)]
    alpha_params_w = [v for k, v in model.named_parameters() if ('alpha' in k and 'alpha_weight' in k)]


    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    # lr = 0.5
    # lr_beta=0.01
    if 'alpha' in tsa_opt:
        params.append({'params': alpha_params})
        # params.append({'params': alpha_params_w, 'lr': 0.2})
    if 'beta' in tsa_opt:
        # params.append({'params': alpha_params_b})
        params.append({'params': beta_params, 'lr': lr_beta})

    # max_iter = 40
    
    optimizer = torch.optim.Adadelta(params, lr=lr) 

    
    if 'alpha' not in tsa_opt:
        with torch.no_grad():
            context_features = model.embed(context_images)
    for i in range(max_iter):
        # if i == max_iter//2:
        #     teacher = copy.deepcopy(model).to(device)
        #     for k, v in teacher.named_parameters():
        #         v.requires_grad=False
        #     # # params.append({'params': alpha_params_b, 'lr': lr_beta})
        #     # params.append({'params': beta_params, 'lr': lr_beta})
        #     optimizer = torch.optim.Adadelta(params, lr=lr)
        #     # model.reset()
        #     # model.eval()

        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in tsa_opt:
            # adapt features by task-specific adapters
            context_features = model.embed(context_images)
        if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
            aligned_features = model.beta(context_features)
        else:
            aligned_features = context_features
            
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)

        # if i >= max_iter//2:
        #     with torch.no_grad():
        #         context_features_t = teacher.embed(context_images)
        #         aligned_features_t = teacher.beta(context_features_t)
        #         context_features_t_o = model.orig_resnet.embed(context_images)
                
        #     loss = 0.7*loss + 0.3*distillation_loss(aligned_features, aligned_features_t, opt=distance)
        # else:
        #     with torch.no_grad():
        #         context_features_t_o = model.orig_resnet.embed(context_images)
        #     loss = 0.9*loss + 0.1*distillation_loss(context_features, context_features_t_o, opt=distance)

        # print(loss.detach().item())
        # 1/

        loss.backward()
        optimizer.step()
    return