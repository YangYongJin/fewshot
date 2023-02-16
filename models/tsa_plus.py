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
# import torch.nn.init as init
import numpy as np
import gc
# import math
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss, distillation_loss, compute_var, DistillKL, prototype_distill_loss
import torch.optim.lr_scheduler as S 
from utils import device
import torchvision



distill = DistillKL(10)

def choose_eff(x):

    if x > 0.75:
        # print("one")
        return 1.5
    else:
        # print("half")
        return 0.5


def make_att(a, b, c=None):
    if c!= None:
        if a.dim() == 0 or a.dim() == 1:
            att_w = F.softmax(torch.FloatTensor([a, b, c]), dim=0)
            att_w = att_w.reshape(1,1,3)
            return att_w.to(a.device)
        else:
            att_w = F.softmax(torch.cat([a, b, c], dim=0), dim=0).T.unsqueeze(0)
            return att_w.to(a.device)
    else:
        if a.dim() == 0 or a.dim() == 1:
            att_w = F.softmax(torch.FloatTensor([a, b]), dim=0)
            att_w = att_w.reshape(2,1)
            return att_w.to(a.device)
        else:
            att_w = F.softmax(torch.cat([a, b], dim=1), dim=1)
            return att_w.to(a.device)


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input>=0.5).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class BasicBlock_tsa(nn.Module):
    def __init__(self, block):
        super(BasicBlock_tsa, self).__init__()
        self.block = copy.deepcopy(block)
        self.conv1 = conv_tsa_up(self.block.conv1)
        planes, in_planes, kernel_size, _ = self.block.conv1.weight.size()
        self.in_planes = in_planes
        self.planes = planes
        self.bn1 =  batch_tsa_up( self.block.bn1 ) 
        self.relu = self.block.relu  
        self.conv2 = conv_tsa_up( self.block.conv2 )  
        self.bn2 = batch_tsa_up(  self.block.bn2 )  
        self.downsample = self.block.downsample  
        self.stride = self.block.stride
        self.padding = self.block.conv2.padding
        self.mode = 1

        self.groups = 8


        self.alpha_norm = nn.Parameter(torch.ones(planes, planes//self.groups, kernel_size, kernel_size), requires_grad=True)
        # self.alpha_norm = nn.Parameter(torch.ones(planes, planes, 1, 1), requires_grad=True)
    
        self.a_bn = nn.BatchNorm2d(planes, eps=1e-05, momentum=1.0, affine=False, track_running_stats=True)


        

        self.relu = nn.ReLU() 




    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out = self.bn1(out1)

        out2 = self.relu(out)         
        out2 = self.conv2(out2) 
        # if self.mode!=1:
        #     out2 = self.conv2(out2) 
        # else:
        #     out2 = self.block.conv2(out2)
        out = self.bn2(out2) 

    
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.mode == 3:
            identity = identity + (F.conv2d(self.a_bn(out1), self.alpha_norm, padding=1, groups=self.groups)) 
            # identity = identity + (F.conv2d(self.a_bn(out1), self.alpha_norm)) 

        elif self.mode == 2:
            identity = identity + (F.conv2d(self.a_bn(out1), self.alpha_norm, padding=1, groups=self.groups)) 

        out = out + identity 
        
        out = self.relu(out)


        return out




def init_batch(m):
    if type(m) == nn.BatchNorm2d and m.momentum == 1.0: 
        m.reset_running_stats()




def train_batch(m):
    if type(m) == nn.BatchNorm2d and m.momentum == 1.0: 
        m.train()

def mode_1(m):
    if hasattr(m,'mode'):
        # print("has it")
        m.mode = 1

def mode_2(m):
    if hasattr(m,'mode'):
        # print("has it")
        m.mode = 2

def mode_3(m):
    if hasattr(m,'mode'):
        # print("has it")
        m.mode = 3

    
class conv_tsa_up(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_up, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, self.kernel_size, kernel_size = self.conv.weight.size()
        self.in_planes = in_planes
        self.mode = 1
        self.groups = 8
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            

            if self.kernel_size == 5:
                self.avgpool = nn.AvgPool2d(7, padding=2, stride=self.conv.stride)
            

            else:
                self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1), requires_grad=True)

                self.avgpool = nn.AvgPool2d(kernel_size, padding=self.conv.padding, stride=self.conv.stride)
                self.maxpool = nn.MaxPool2d(kernel_size, padding=self.conv.padding, stride=self.conv.stride)
    
                
    def forward(self, x):
        y = self.conv(x)
        if self.kernel_size == 5:
            if not self.mode == 1:
                pass
                # y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            else:
                pass
                # y = F.conv2d(y, self.alpha_weight)
        else:
            if self.mode==1:
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride) 
                # y = y + F.conv2d(self.avgpool(x), self.alpha) 
            elif self.mode==3:
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
                # y = y + F.conv2d(self.avgpool(x), self.alpha) 
            # elif self.mode==2:
                # y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            #     y = y + F.conv2d(x, self.alpha_norm2, padding=self.conv.padding, stride=self.conv.stride, groups=self.groups) 

                     
        return y

class batch_tsa_up(nn.Module):
    def __init__(self, orig_batch):
        super(batch_tsa_up, self).__init__()
        # the original conv layer
        self.batch = copy.deepcopy(orig_batch)
        self.batch.weight.requires_grad = False
        self.batch.bias.requires_grad = False
        self.num_features = self.batch.weight.size()[0]
        self.groups = 32

        self.mode = 2
        self.relu = nn.ReLU()

        self.a_bn = nn.BatchNorm2d(self.num_features, eps=1e-05, momentum=1.0, affine=False, track_running_stats=True)
        self.alpha_norm = nn.Parameter(torch.ones(self.num_features, self.num_features, 1, 1), requires_grad=True)


    def forward(self, x):
        y = self.batch(x)


        return y



class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        # self.beta_weight = nn.Parameter(torch.ones(1,feat_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim),    requires_grad = True)

        self.feat_dim = feat_dim
        
        self.layer_norm = nn.LayerNorm(feat_dim, elementwise_affine=False)

        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        if len(list(x.size())) == 2:
            # print(x.var(1))
            identity = x
            x_norm = self.layer_norm(x)
            # x = x_norm #.unsqueeze(-1).unsqueeze(-1)
            x = x.unsqueeze(-1).unsqueeze(-1)

        else:
            x = x.flatten(1)
            # print(x.var(1))
            identity = x
            x_norm = self.layer_norm(x)
            # x = x_norm #.unsqueeze(-1).unsqueeze(-1)
            x = x.unsqueeze(-1).unsqueeze(-1)

        x = (F.conv2d(x, self.weight.to(x.device))).flatten(1)


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
            
        #     if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 5:
        #         new_conv = conv_tsa_up(m)
        #         setattr(orig_resnet, name, new_conv)

        #     elif isinstance(m, nn.BatchNorm2d):
        #         new_batch = batch_tsa_up(m)
        #         setattr(orig_resnet, name, new_batch)

    
        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for i, block in enumerate(orig_resnet.layer1):
            orig_resnet.layer1[i] = BasicBlock_tsa(block)
        for i, block in enumerate(orig_resnet.layer2):
            orig_resnet.layer2[i] = BasicBlock_tsa(block)
        for i, block in enumerate(orig_resnet.layer3):
            orig_resnet.layer3[i] = BasicBlock_tsa(block)
        for i, block in enumerate(orig_resnet.layer4):
            orig_resnet.layer4[i] = BasicBlock_tsa(block)
        

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta_low = pa(feat_dim)
        beta_mid = pa(feat_dim)
        beta_high = pa(feat_dim)
        beta_orig = pa(feat_dim)
        setattr(self, 'beta_low', beta_low)
        setattr(self, 'beta_mid', beta_mid)
        setattr(self, 'beta_high', beta_high)
        setattr(self, 'beta_orig', beta_orig)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

        

    def embed_concat(self, x, c_features=None):
        with torch.no_grad():
            context_features_t_o = self.orig_resnet.embed(x)
            # self.apply(false_normalize)
            # context_features_low = self.embed(x)
            self.apply(mode_3)
            context_features_high = self.embed(x)

            # b, c = context_features_low.size()
        
            
        if 'beta' in args['test.tsa_opt']:
            with torch.no_grad():
                # aligned_features_low = self.beta_low(context_features_low)
                aligned_features_high = self.beta_high(context_features_high)
                # aligned_features_orig = self.beta_orig(context_features_t_o)


                t_features = aligned_features_high
                

        else:
            t_features = context_features_t_o


        
        return t_features

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        
        for k, v in self.named_parameters():
            if 'alpha' in k and v.requires_grad:
                # nn.init.constant_(v, 0.0)
                if 'bias' not in k:
                    if 'eff' in k:
                        v.data = torch.ones(v.size()).to(v.device)#*1e-4
                    elif 'weight' in k:
                        # v.data = torch.ones(v.size()).to(v.device)#*1e-3
                        v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                      
                    else:
                        if 'norm' in k:
                            v.data = (torch.rand(v.size()).to(v.device))*0.0001

                        else:
                            v.data =  torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)*0.0001

                else:
                    v.data = torch.zeros(v.size()).to(v.device)

            elif 'beta' in k and v.requires_grad:

                # initialize pre-classifier alignment mapping (beta)
                if 'bias' in k:
                    v.data = torch.zeros(v.size()).to(v.device)
                else:
                    v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
        self.apply(init_batch)



def train_adapter(optimizer, scheduler, model, x, y, tsa_opt, max_iter, distance, beta='low', c_features = None, eff=0, scale=1.0, fixed=False, c_features2 = None, eff2=0):
    
    n_way = len(y.unique())
    a=0
    n_shot = x.size(0)//n_way
    # max_iter = min(max(n_shot, n_way,20),50)
    # print(max_iter)
    iter = 100
    # print(beta)
    if fixed:
        iter = max_iter
    for i in range(iter):
            

        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in tsa_opt and beta != 'orig':
            # adapt features by task-specific adapters
            context_features = model.embed(x)

        if beta == 'low':
            aligned_features = model.beta_low(context_features)
        elif beta=='high':
            aligned_features = model.beta_high(context_features)
        elif beta=='mid':
            aligned_features = model.beta_mid(context_features)
        elif beta=='orig':
            aligned_features = model.beta_orig(x)
        else:
            aligned_features = context_features

        

            
        loss, stat, _ = prototype_loss(aligned_features, y,
                                       aligned_features, y, distance=distance)

        # if fixed:
        #     print(i,  n_way, n_shot, loss, stat['acc'])



        loss = loss * scale
        # print('b',eff)
        # eff = min(max(torch.randn(1).cuda()*0.01 + eff, 0.0), 1.0)
        # print('a',eff)
       
        if c_features != None:
            distill_loss = distillation_loss(aligned_features, c_features, opt=distance) 
            
            loss = (1-eff)*loss + (eff)* distill_loss 
            
        if  c_features != None and c_features2 != None:
            loss = (1-eff2)*loss + eff2* distillation_loss(aligned_features, c_features2, opt='cos')

        
        
        loss.backward()
        optimizer.step()
        if scheduler!=None:
            scheduler.step()

        if stat['acc'] > 0.99 and not fixed:
            a += 1
            if a > max_iter:
                # if c_features != None:
                #     print(loss, distill_loss)
                break

def train_one_set(model, max_iter, lr, lr_w, lr_beta, tsa_opt, x, y, distance, beta='low', reset=False, c_features = None, eff=0, scale=1.0, fixed=False, c_features2 = None, eff2 = 0):

    alpha_params = [v for k, v in model.named_parameters() if ('alpha' in k and ('alpha_weight' not in k and 'alpha_bias' not in k))]
    alpha_params_w = [v for k, v in model.named_parameters() if ('alpha' in k and ('alpha_weight' in k or 'alpha_bias' in k)) or 'a_bn' in k]


    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    
     
    if 'alpha' in tsa_opt:
        params.append({'params': alpha_params})
        params.append({'params': alpha_params_w, 'lr': lr_w})
    if 'beta' in tsa_opt:
        params.append({'params': beta_params, 'lr': lr_beta})
    
    optimizer = torch.optim.Adadelta(params, lr=lr)
    scheduler=None
    # scheduler = S.CosineAnnealingLR(optimizer, max_iter)

    if beta == 'orig':
        train_adapter(optimizer, scheduler, model, x, y, tsa_opt, max_iter, distance, beta=beta, c_features = c_features, eff=eff, scale=scale, fixed=fixed, c_features2=c_features2, eff2=eff2)
        with torch.no_grad():
            aligned_features_t_o = model.beta_orig(x)
    elif beta=='high':
        model.apply(mode_3)
        train_adapter(optimizer, scheduler, model, x, y, tsa_opt, max_iter, distance, beta=beta, c_features = c_features, eff=eff, scale=scale, fixed=fixed, c_features2=c_features2, eff2=eff2)
        with torch.no_grad():
            aligned_features_t_o = model.beta_high(model.embed(x))
    elif beta=='mid':
        model.apply(mode_2)
        train_adapter(optimizer, scheduler, model, x, y, tsa_opt, max_iter, distance, beta=beta, c_features = c_features, eff=eff, scale=scale, fixed=fixed, c_features2=c_features2, eff2=eff2)
        with torch.no_grad():
            aligned_features_t_o = model.beta_mid(model.embed(x))
            
    elif beta=='low':
        model.apply(mode_1)
        train_adapter(optimizer, scheduler, model, x, y, tsa_opt, max_iter, distance, beta=beta, c_features = c_features, eff=eff, scale=scale, fixed=fixed, c_features2=c_features2, eff2=eff2)
        with torch.no_grad():
            aligned_features_t_o = model.beta_low(model.embed(x))

    if reset:
        model.reset()



    return aligned_features_t_o



def tsa_plus(context_images, context_labels, model, max_iter=40, scale=0.1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    c_features = None
    e_features = None

    model.eval()
    model.apply(train_batch)


    with torch.no_grad():
        e_features = model.orig_resnet.embed(context_images)
        loss_t_o, stat_t_o, _ = prototype_loss(e_features, context_labels,
                                       e_features, context_labels, distance=distance)
    
    tsa_opt = args['test.tsa_opt']

    lr_w = 0.1*scale
    
    lr = 0.5*scale
    lr_beta = 1.0*scale

    n_way = len(context_labels.unique())
    b_size = context_labels.size(0)
    n_shot = b_size//n_way
    betas = ['low']
    intra_sim, inter_sim, whole_sim = compute_var(e_features, context_labels, n_way)
    lrs = torch.Tensor([lr, lr, lr])
    lr_betas = torch.Tensor([lr_beta, lr_beta, lr_beta])

    eff_bias = 0.75
     
    for i in range(len(betas)):
        x = context_images
        
        if betas[i] == 'orig':
            x = e_features


        c_features = train_one_set(model, max_iter=10, lr=lrs[i], lr_w=lr_w, lr_beta=lr_betas[i], tsa_opt=tsa_opt, x=x, y=context_labels, distance=distance, beta=betas[i], reset=True, c_features = None, eff=0.0, scale=1.0, fixed=False)

        
        sim = 1.0-torch.abs(inter_sim-intra_sim)*2
        eff = min(torch.tanh((sim)*eff_bias), 1.0)  

        e_features = eff * e_features + (1.0-eff) * c_features

        intra_sim, inter_sim, _ = compute_var(e_features, context_labels, n_way)

    sim = 1.0-torch.abs(inter_sim-intra_sim)
    eff = min(torch.tanh((sim*eff_bias)), 1.0)
    # print(whole_sim)
    c_features = train_one_set(model, max_iter=15, lr=lr*0.5, lr_w=lr_w, lr_beta=lr_beta, tsa_opt=tsa_opt, x=context_images, y=context_labels, distance=distance, beta='high', reset=False, c_features = e_features, eff = eff, scale=1.0, fixed=False, c_features2= None, eff2 = eff)


    model.eval()

    return (lr, lr_beta, lr_w,eff_bias, c_features)