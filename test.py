import torch
import torch.nn as nn
import numpy as np


x = torch.arange(20).reshape(5,4).float()
x = torch.FloatTensor([
    [5,4,7,2,3],
    [3,4,2,1,6]

])
_, feat_dim = x.size()
topk, indices = torch.topk(x, feat_dim*4//5, 1, False, False)
lowk, lowindices = torch.topk(x, feat_dim*4//5, 1, True, False)
res = torch.zeros(x.size()).to(x.device)
# res.requires_grad=True
res1 = res.scatter(1, indices, topk)
res2 = res.scatter(1, lowindices, lowk)

res = x*(res1!=0)*(res2!=0)

print(res1!=0)
print(res2!=0)
print(res)