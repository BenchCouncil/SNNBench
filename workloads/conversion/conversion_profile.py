import os
import argparse
import random
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.profiler import profile, record_function, tensorboard_trace_handler, ProfilerActivity

# from minibatch import ROOT_DIR
from mlp import MLP

random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker,
#     generator=g,
# )

def trace_handler(prof):
    # print(vars(prof))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=-1))
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    # with open(f'profile_${MODEL}.pkl', 'wb') as outp:
    #     pickle.dump(prof, outp)
    sys.exit(0)

def main(args):
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    device = torch.device("cuda" if args.gpu else "cpu")

    # Get the MNIST data.
    kwargs = {"num_workers": 1, "pin_memory": True} if args.gpu else {}
    train_dataset = MNIST(
        os.path.join('.', "data"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_dataset = MNIST(
        os.path.join('.', "data"),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Create and train an ANN on the MNIST dataset.
    ann = MLP().to(device)

    # Specify optimizer and loss function.
    optimizer = optim.Adam(params=ann.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Train / test the ANN.
    best_accuracy = -np.inf
    for epoch in range(1, args.n_epochs + 1):

        # Training.
        ann.train()
        correct = 0
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    # torch.profiler.ProfilerActivity.CUDA,
                    ],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=100,
                    repeat=1),
                # on_trace_ready=tensorboard_trace_handler(LOG),
                on_trace_ready=trace_handler,
                with_stack=False
                ) as profiler:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = ann(data)
                loss = criterion(output, target)

                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()
                optimizer.step()
                profiler.step()
                if batch_idx % 10 == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

        print(
            "\nTrain accuracy: {:.2f}%".format(
                100.0 * correct / len(train_loader.dataset)
            )
        )

        # Testing.
        ann.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass.
                output = ann(data)

                # Sum batch loss.
                test_loss += criterion(output, target).item()

                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100.0 * correct / len(test_loader.dataset)
        if accuracy > best_accuracy:
            # Save model to disk.
            f = os.path.join(args.job_dir, "ann.pt")
            os.makedirs(os.path.dirname(f), exist_ok=True)
            torch.save(ann.state_dict(), f=f)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), accuracy
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
