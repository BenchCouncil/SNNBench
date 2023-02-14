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
from torch.profiler import profile, record_function, tensorboard_trace_handler, ProfilerActivity

import matplotlib.pyplot as plt
import numpy as np
import itertools

import random
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_interop_threads", type=int)
parser.add_argument("--num_threads", type=int)

args = parser.parse_args()

num_interop_threads = args.num_interop_threads
num_threads = args.num_threads

if num_interop_threads is not None:
    torch.set_num_interop_threads(num_interop_threads)
if num_threads is not None:
    torch.set_num_threads(num_threads)
print(torch.__config__.parallel_info())

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

net = nn.Sequential(nn.Conv2d(1, 12, 5),
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

def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=-1))
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    sys.exit(0)

def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    # torch.profiler.ProfilerActivity.CUDA,
                    ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=10,
                    repeat=10),
                # on_trace_ready=tensorboard_trace_handler(LOG),
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=False,
                ) as profiler:
            for data, targets in train_loader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec, _ = forward_pass(net, num_steps, data)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

                profiler.step()

    return acc / total


test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"The total accuracy on the test set is: {test_acc * 100: .2f}%")
sys.exit(0)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.9999))
num_epochs = 10
test_acc_hist = []


with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
            ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=1,
            repeat=1),
        # on_trace_ready=tensorboard_trace_handler(LOG),
        on_trace_ready=trace_handler,
        with_stack=True
        ) as profiler:
    for epoch in range(num_epochs):
        avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                num_steps=num_steps, time_var=False, device=device)
        print(f"Epoch {epoch}, Train Loss: {avg_loss.item(): .2f}")
        profiler.step()

        # test_acc = batch_accuracy(test_loader, net, num_steps)
        # test_acc_hist.append(test_acc)
        # print(f"Epoch {epoch}, Test Acc: {test_acc * 100: .2f}%\n")
