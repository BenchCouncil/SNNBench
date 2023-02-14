import torch

import os

# torch.set_num_interop_threads(6)
# torch.set_num_threads(6)
print(f'cpu count: {os.cpu_count()}')
print(torch.__config__.parallel_info())

