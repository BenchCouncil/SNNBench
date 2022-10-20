import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
# from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

import random
import os
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

batch_size = 128
data_path = './data/mnist'
subset = 10

dtype = torch.float
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running on {device}')

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
    ])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# utils.data_subset(mnist_train, subset)
# utils.data_subset(mnist_test, subset)

# train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker, generator=g)

net = nn.Sequential(
        nn.Conv2d(1, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 64, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(64*4*4, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        ).to(device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)

    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)
print(f"The loss from an untrained network is {loss_val.item():.3f}")

acc = SF.accuracy_rate(spk_rec, targets)


def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


# print(f'test_loader size: {len(test_loader)}')
test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"The total accuracy on the test set is: {test_acc * 100: .2f}%")

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.9999))
num_epochs = 10
test_acc_hist = []

for epoch in range(num_epochs):
    # print(f'In training phase, train_loader size: {len(train_loader)}')
    start = perf_counter()
    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
            num_steps=num_steps, time_var=False, device=device)
    print(f'Training elapsed time: {perf_counter() - start}')
    print(f"Epoch {epoch}, Train Loss: {avg_loss.item(): .2f}")

    # print(f'In training phase, test_loader size: {len(test_loader)}')
    start = perf_counter()
    test_acc = batch_accuracy(test_loader, net, num_steps)
    print(f'Inference elapsed time: {perf_counter() - start}')
    test_acc_hist.append(test_acc)
    print(f"Epoch {epoch}, Test Acc: {test_acc * 100: .2f}%\n")
