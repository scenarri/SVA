from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import argparse
from pprint import pprint
from tqdm import tqdm

from utils import *



def main(args):



    seed_everything(7)
    lce = nn.CrossEntropyLoss()

    val_dataset = datasets.ImageFolder("./dataset/testset")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    val_num = len(val_dataset)

    victim_model_names = ['aconv', 'alex', 'vgg', 'dense', 'resnet', 'resnext', 'incres', 'incv4']
    victim_models = [WrapperModel(load_models(x), size=inputsize(x)) for x in victim_model_names]
    ASR = {m: 0. for m in victim_model_names}

    surrogate = WrapperModel(load_models(args.surrogate), size=inputsize(args.surrogate))

    eps = args.eps
    if args.baseline == 'FGSM':
        total_iter, alpha = 1, eps
    elif args.baseline == 'BIM':
        total_iter, alpha = 20, eps/10
    elif args.baseline == 'PGD':
        total_iter, alpha = 50, eps/10

    kernel = gkern(15, 3).astype(np.float32)
    stack_kernel = np.expand_dims(kernel, 0)
    stack_kernel = np.expand_dims(stack_kernel, 0)
    Tikernel = torch.from_numpy(stack_kernel).cuda()

    loader = tqdm(val_loader)
    for iters, (imgseg, target) in enumerate(loader):
        data, seg = batch_center_crop(imgseg, 88, 88)
        data, seg, target = data.cuda(), seg.cuda(), target.cuda()

        data = Resize(data, inputsize(args.surrogate))
        seg = Resize(seg, inputsize(args.surrogate))

        dataori = data.detach().clone()

        momentum = torch.zeros_like(data)
        delta = torch.zeros_like(data)

        if args.baseline == 'PGD':
            data = data + torch.empty_like(data).uniform_(-eps, eps) * seg
            data = torch.clamp(data, min=0, max=1).detach()

        for attak_iters in range(total_iter):
            data.requires_grad = True

            if args.attack == 'NA':
                output = surrogate(data)
                celoss = lce(output, target)
                grad = torch.autograd.grad(celoss, data, create_graph=False, retain_graph=False)[0]
            elif args.attack == 'DI':
                output = surrogate(input_diversity(data, prob=0.6, rsz=np.ceil(1.1*inputsize(args.surrogate)), orisize=inputsize(args.surrogate)))
                celoss = lce(output, target)
                grad = torch.autograd.grad(celoss, data, create_graph=False, retain_graph=False)[0]
            elif args.attack == 'SVA':
                output = surrogate(specklevariant(data, beta=args.beta, data_size=inputsize(args.surrogate), kernel_size=args.s))
                celoss = lce(output, target)
                grad = torch.autograd.grad(celoss, data, create_graph=False, retain_graph=False)[0]
            elif args.attack == 'SI':
                grad = 0
                for Scaleiter in range(5):
                    scale_factor = 2 ** (-Scaleiter)
                    datacopy = scale_factor * data
                    output = surrogate(datacopy)
                    celoss = lce(output, target)
                    grad += torch.autograd.grad(celoss, data, create_graph=False, retain_graph=False)[0]
                grad = grad / 5

            grad = F.conv2d(grad, Tikernel, bias=None, stride=1, padding='same', groups=1) #10
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * 1.
            momentum = grad
            delta = delta.detach() + alpha * grad.sign() * seg
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = (torch.clamp(dataori + delta, min=0, max=1) - dataori).detach()
            data = torch.clamp(dataori + delta, min=0, max=1).detach()

        advdata = Resize(data, 88)

        with torch.no_grad():
            for m in range(len(victim_models)):
                ASR[victim_model_names[m]] += (torch.sum(victim_models[m](advdata).argmax(1) != target).float().item() / val_num * 100)

    avg = 0
    transfer = 0
    for m in range(len(victim_model_names)):
        avg += ASR[victim_model_names[m]]
        if victim_model_names[m] != args.surrogate:
            transfer += ASR[victim_model_names[m]]
        ASR[victim_model_names[m]] = round(ASR[victim_model_names[m]], 2)
    avg /= len(victim_model_names)
    transfer /= (len(victim_model_names)-1)
    pprint(ASR, sort_dicts=False)
    print('Average ASR: {}%'.format(round(avg, 2)))
    print('Average Transfer: {}%'.format(round(transfer, 2)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eps', type=float, default=48/255, help='l_\infty-norm perturbation budget')
    parser.add_argument('--baseline', type=str, default='BIM', help='baseline including FGSM/BIM/PGD')
    parser.add_argument('--attack', type=str, default='SVA', help='attack including NA/DI/SI/SVA')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.5, help='tail for truncated exponential distribution')
    parser.add_argument('--s', type=int, default=7, help='kernel size for median filtering')
    parser.add_argument('--surrogate', type=str, default='vgg')

    main(parser.parse_args())


