"""
Helpers for distributed training.
"""


import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).


def setup_dist():
    """
    Setup a distributed process group.
    """
    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")
  