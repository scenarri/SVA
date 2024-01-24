import torch
import numpy as np
import os
import cv2
import scipy.signal.windows as wind
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from models.arch import AConvNets, alexnet, densenet, inceptionv4, resnet, inception_resnet_v2, vgg, utils
import random
import kornia
import scipy.stats as stats

def seed_everything(seed):
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def input_diversity(data, prob=0.5, rsz=248, orisize=224):
    if float(torch.rand([1, 1])) > 1 - prob:
        rnd = int(torch.rand([1, 1]) * (rsz - orisize)) + orisize
        h_rem = rsz - rnd
        w_rem = rsz - rnd
        pad_top = int(torch.rand([1, 1]) * h_rem)
        pad_bottom = int(h_rem - pad_top)
        pad_left = int(torch.rand([1, 1]) * w_rem)
        pad_right = int(w_rem - pad_left)
        data = F.interpolate(data, size=rnd, mode='nearest')
        data = F.pad(data, (pad_left, pad_right, pad_top, pad_bottom))
        data = F.interpolate(data, size=orisize, mode='nearest')
        return data
    else:
        return data


def specklevariant(data, beta, data_size, kernel_size):
    X = stats.truncexpon(b=beta, loc=0, scale=1)
    inter2 = kornia.filters.median_blur(data, (kernel_size, kernel_size))
    noise = X.rvs([data_size, data_size])
    output = inter2 * torch.from_numpy(noise).cuda().float()
    output = torch.clamp(output, min=0, max=1)
    return output

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def inputsize(name):
    size = 224
    if name == 'aconv':
        size = 88
    elif 'inc' in name:
        size = 299
    return size

def showimg(data, line=0):
    img = data.detach().cpu().numpy()
    plt.imshow(img[line][0], vmin=0, vmax=1, cmap='gray')
    plt.show()

def batch_center_crop(img,cropx=88,cropy=88):
    data = torch.zeros([img.size()[0], 1, cropx, cropy])
    seg = torch.zeros([img.size()[0], 1, cropx, cropy])
    _, _, y, x = img.shape
    for i in range(img.size()[0]):
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        data[i][0] = img[i][0][starty:starty + cropy, startx:startx + cropx]
        seg[i][0] = img[i][1][starty:starty + cropy, startx:startx + cropx]
    return data, seg

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def load_models(model_name):
    if model_name == 'aconv':
        net = AConvNets.AConvNets()
        net.load_state_dict(torch.load('./models/weights/aconv.pth'))
    elif model_name == 'alex':
        net = alexnet.alexnet()
        net.load_state_dict(torch.load('./models/weights/alexnet.pth'))
    elif model_name == 'vgg':
        net = vgg.vgg16()
        net.load_state_dict(torch.load('./models/weights/vgg16.pth'))
    elif model_name == 'dense':
        net = densenet.densenet121()
        net.load_state_dict(torch.load('./models/weights/densenet121.pth'))
    elif model_name == 'resnet':
        net = resnet.resnet50()
        net.load_state_dict(torch.load('./models/weights/resnet50.pth'))
    elif model_name == 'resnext':
        net = resnet.resnext50_32x4d()
        net.load_state_dict(torch.load('./models/weights/resnext50.pth'))
    elif model_name == 'incres':
        net = inception_resnet_v2.Inception_ResNetv2()
        net.load_state_dict(torch.load('./models/weights/incresv2.pth'))
    elif model_name == 'incv4':
        net = inceptionv4.Inceptionv4()
        net.load_state_dict(torch.load('./models/weights/incv4.pth'))

    return net.eval().cuda()

def Resize(img, size):
    img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return img

class WrapperModel(nn.Module):
    def __init__(self, model, size, resize=True):
        super(WrapperModel, self).__init__()
        self.model = model
        self.resize = resize
        self.size = size
    def forward(self, x):
        if self.resize == True:
            x = self.Resize(x, self.size)
        return self.model(x)

    def Resize(self, img, size):
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        return img