from speech_commands.model import LSTMModel, lsnn_model, lif_model
from speech_commands.speech_commands import SpeechCommandsDataset, prepare_dataset

import torch
from torch.profiler import profile, record_function, tensorboard_trace_handler, ProfilerActivity
import torchaudio
import numpy

import os
import sys
import time
import random
import pickle
import argparse

# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()

print(torch.__config__.parallel_info())
# sys.exit()

torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--batch_size", default=16)
parser.add_argument("--device", default="cuda")
parser.add_argument("--model", default="lif")
parser.add_argument("--log", default=".")

args = parser.parse_args()

BATCH_SIZE = args.batch_size  # 16
LR = args.learning_rate  # 0.0001
DEVICE = args.device  # "cuda"
MODEL = args.model  # "lif"
LOG = args.log


class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


speech_commands = torchaudio.datasets.SPEECHCOMMANDS(root=".", download=True)
train_sc = SubsetSC('training')
valid_sc = SubsetSC('validation')
test_sc = SubsetSC('testing')
# train_sc, valid_sc, test_sc = prepare_dataset(speech_commands)

print(f'train_sc: {len(train_sc)}, valid_sc: {len(valid_sc)}, test_sc: {len(test_sc)}')


train_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
)
test_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
)

train_speech_commands = SpeechCommandsDataset(
    dataset=train_sc, transform=train_transform
)
valid_speech_commands = SpeechCommandsDataset(
    dataset=valid_sc, transform=valid_transform
)
test_speech_commands = SpeechCommandsDataset(
    dataset=test_sc, transform=test_transform
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# pytype: disable=module-attr
train_loader = torch.utils.data.DataLoader(
    train_speech_commands, batch_size=BATCH_SIZE, shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

valid_loader = torch.utils.data.DataLoader(
    valid_speech_commands, batch_size=BATCH_SIZE, shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader = torch.utils.data.DataLoader(
    test_speech_commands, batch_size=BATCH_SIZE, shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
# pytype: enable=module-attr

if MODEL == "lif":
    model = lif_model(n_output=13).to(DEVICE)
elif MODEL == "lsnn":
    model = lsnn_model(n_output=13).to(DEVICE)
else:
    model = LSTMModel(n_output=13).to(DEVICE)

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def trace_handler(prof):
    # print(vars(prof))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=-1))
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    # with open(f'profile_${MODEL}.pkl', 'wb') as outp:
    #     pickle.dump(prof, outp)
    sys.exit(0)

def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
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
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                data = data.squeeze(1)
                data = data.permute(2, 0, 1)
                output = model(data)
                test_loss += torch.nn.functional.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                profiler.step()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)"
    )

    return test_loss, accuracy


test(model, test_loader, 0)
# exit(0)

for epoch in range(1):
    model.train()
    print(f"=========== {epoch} ===========")
    start = time.perf_counter()

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
        for idx, (data, target) in enumerate(train_loader):
            data = data.squeeze(1)
            data = data.permute(2, 0, 1)
            model.zero_grad()
            out = model(data.to(DEVICE))
            loss = loss_function(out, target.to(DEVICE))
            loss.backward()
            optimizer.step()
            profiler.step()
            if idx % 1000 == 0:
                print(f"{idx} {loss.data}")
    # with torch.autograd.profiler.emit_nvtx():
    #     for idx, (data, target) in enumerate(train_loader):
    #         data = data.squeeze(1)
    #         data = data.permute(2, 0, 1)
    #         model.zero_grad()
    #         out = model(data.to(DEVICE))
    #         loss = loss_function(out, target.to(DEVICE))
    #         loss.backward()
    #         optimizer.step()
    #         if idx % 1000 == 0:
    #             print(f"{idx} {loss.data}")
    end = time.perf_counter()
    print(f'Training time of one epoch: {end - start:0.4f} seconds')

    start = time.perf_counter()
    # test(model, valid_loader, epoch)
    test(model, test_loader, epoch)
    end = time.perf_counter()
    print(f'Inference time: {end - start:0.4f} seconds')
