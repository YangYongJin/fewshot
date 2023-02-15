class conv_tsa_deform(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_deform, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, kernel_size, kernel_size = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = DeformableConv2d(in_channels=in_planes, out_channels=planes, kernel_size=kernel_size, stride=self.conv.stride, padding=self.conv.padding)
            self.alpha.regular_conv.data = self.conv.weight.data
            self.alpha.regular_conv.requires_grad = False

    def forward(self, x):
        y = self.conv(x)
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            y = self.alpha(x) #F.conv2d(x, (self.alpha * self.conv.weight), stride=self.conv.stride, padding=self.conv.padding)
        return y


class conv_tsa_se(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_se, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, kernel_size, kernel_size = self.conv.weight.size()
        self.planes = planes
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            reduction = 2
            self.alpha = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            b, c, _, _ = x.size()
            v = F.adaptive_avg_pool2d(x,1).view(b,c)
            x = x + x * self.alpha(v).view(b,c,1,1).expand_as(x) #F.conv2d(x, (self.alpha * self.conv.weight), stride=self.conv.stride, padding=self.conv.padding)
            y = self.conv(x)
        else:
            y = self.conv(x)
        return y

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input>0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
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
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class conv_tsa_add(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_add, self).__init__()
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
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class conv_tsa_group(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_group, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.group=8
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = nn.Parameter(torch.ones(planes, in_planes//self.group, 1, 1))
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride, groups=self.group)
        return y

class conv_tsa_dynamic(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa_dynamic, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, kernel_size, kernel_size = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
        
            self.alpha_weight = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            self.alpha_weight.requires_grad = True

    def forward(self, x):
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            y = F.conv2d(x, self.conv.weight*self.alpha_weight, padding=self.conv.padding, stride=self.conv.stride)
        else:
            y = self.conv(x)
        return y

class batch_tsa(nn.Module):
    def __init__(self, orig_batch):
        super(batch_tsa, self).__init__()
        # the original conv layer
        self.batch = copy.deepcopy(orig_batch)
        self.batch.weight.requires_grad = False
        self.batch.bias.requires_grad = False
        self.num_features = self.batch.weight.size()[0]
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = nn.Parameter(torch.ones(self.num_features, self.num_features,1 ,1))

            # self.alpha_weights.requires_grad = True
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.batch(x)
        if 'alpha' in args['test.tsa_opt']:
            y = y * F.conv2d(x, self.alpha)

        return y

def make_dstill(aligned_features_high, aligned_features_low, aligned_features_orig):
    middle_point = (aligned_features_orig+aligned_features_high+aligned_features_low)/3

    distill_low = distillation_loss(middle_point, aligned_features_low, opt='cos', reduce=False)
    distill_high = distillation_loss(middle_point, aligned_features_high, opt='cos', reduce=False)
    distill_origin = distillation_loss(aligned_features_high, aligned_features_low, opt='cos', reduce=False)
    print(distill_high.size())

    return (distill_high,distill_low, distill_origin)

def make_dstill2(aligned_features_high, aligned_features_low, aligned_features_orig, context_features):

    distill_low = distillation_loss(context_features, aligned_features_low, opt='cos', reduce=True)
    distill_high = distillation_loss(context_features, aligned_features_high, opt='cos', reduce=True)
    distill_origin = distillation_loss(context_features, aligned_features_orig, opt='cos', reduce=True)

    return (distill_high,distill_low, distill_origin)

def make_dstill3(aligned_features_high, aligned_features_low, aligned_features_orig):

    distill_high= distillation_loss(aligned_features_high, aligned_features_orig, opt='cos', reduce=True)
    distill_low = distillation_loss(aligned_features_low, aligned_features_orig, opt='cos', reduce=True)
    distill_diff = distillation_loss(aligned_features_high, aligned_features_low, opt='cos', reduce=True)

    return (distill_high,distill_low, distill_diff)

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)

def top_k(x):
    _, feat_dim = x.size()
    topk, indices = torch.topk(torch.abs(x), feat_dim*9//10, 1, False)
    res = torch.zeros(x.size()).to(x.device)
    # res.requires_grad=True
    res = res.scatter(1, indices, topk)

    return res

def mini(x):
    return x[1]


sorted(a, key = mini)