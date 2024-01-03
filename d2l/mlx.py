import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import data as dx
from mlx.data.datasets.common import CACHE_DIR, ensure_exists, urlretrieve_with_progress

nn_Module = nn.Module


import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

import gzip
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile



def use_svg_display():
    """使用svg格式在Jupyter中显示绘图

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)



def sgd(params, grads, lr, batch_size):
    """小批量随机梯度下降
    Defined in :numref:`sec_linear_scratch`"""
    newparams = []
    for param, grad in zip(params, grads):
        # print(param)
        newparams.append(param - lr * grad)
        
        # mx.eval(param)

    return newparams


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))



def load_data_fashion_mnist(batch_size, resize=None):
    """Load a buffer with the MNIST dataset.

    If the data doesn't exist download it and save it for the next time.

    Args:
        root (Path or str, optional): The directory to load/save the data. If
            none is given the ``~/.cache/mlx.data/mnist`` is used.
        train (bool): Load the training or test set.
    """
   
    root = CACHE_DIR / "mnist"


    ensure_exists(root)

    def download():
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        filename = [
            [NamedTemporaryFile(), "training_images", "train-images-idx3-ubyte.gz"],
            [NamedTemporaryFile(), "test_images", "t10k-images-idx3-ubyte.gz"],
            [NamedTemporaryFile(), "training_labels", "train-labels-idx1-ubyte.gz"],
            [NamedTemporaryFile(), "test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for out_file, _, name in filename:
            urlretrieve_with_progress(base_url + name, out_file.name)

        for out_file, key, _ in filename[:2]:
            with gzip.open(out_file.name, "rb") as f:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28, 28, 1
                )
        for out_file, key, _ in filename[-2:]:
            with gzip.open(out_file.name, "rb") as f:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)
        train_set = [
            {"image": mnist["training_images"][i], "label": mnist["training_labels"][i]}
            for i in range(len(mnist["training_images"]))
        ]
        test_set = [
            {"image": mnist["test_images"][i], "label": mnist["test_labels"][i]}
            for i in range(len(mnist["test_images"]))
        ]

        with (root / "train.pkl").open("wb") as f:
            pickle.dump(train_set, f)
        with (root / "test.pkl").open("wb") as f:
            pickle.dump(test_set, f)

    if not (root / "test.pkl").is_file():
        download()

    train_file = (root / "train.pkl") 
    test_file =  (root / "test.pkl")

    train_iter = []
    test_iter = []
    with train_file.open("rb") as f:
        train_stream = dx.buffer_from_vector(pickle.load(f))\
            .shuffle()\
            .to_stream()\
            .key_transform("image", lambda x: x.astype("float32").reshape(-1))\
            .key_transform("image", lambda x: x / 255)\
            .batch(batch_size)
        for batch in train_stream:
            train_iter.append(batch)

    with test_file.open("rb") as f:
        test_stream = dx.buffer_from_vector(pickle.load(f))\
            .shuffle()\
            .to_stream()\
            .key_transform("image", lambda x: x.astype("float32").reshape(-1))\
            .key_transform("image", lambda x: x / 255)\
            .batch(batch_size)
        for batch in test_stream:
            test_iter.append(batch)

    return train_iter, test_iter
        
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def accuracy(y_hat, y) -> float:  #@save
#     """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return cmp.astype(y.dtype).sum().astype(y_hat.dtype).item()
    
def evaluate_accuracy(net, data_iter)-> float:  #@save
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for batch in data_iter:
        X, y = mx.array(batch["image"]), mx.array(batch["label"])
        metric.add(accuracy(net(X), mx.array(y)), y.size)

    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    loss_fn = lambda model, X, y : mx.mean(loss(model(X), y))
    # def loss_fun(model, X, y):

    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
    for batch in train_iter:
        # 计算梯度并更新参数
        X, y = mx.array(batch["image"]), mx.array(batch["label"])
        lvalue, grads = loss_and_grad_fn(net, X, y)
        updater.update(net, grads)
        metric.add(lvalue.item(), accuracy(net(X), y), y.size)
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
     
        train_metrics = train_epoch_ch3(net,  train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
      
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
