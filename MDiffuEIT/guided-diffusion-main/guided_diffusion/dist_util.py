"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    #comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
    # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # os.environ["RANK"] = str(comm.rank)
    # os.environ["WORLD_SIZE"] = str(comm.size)

    dist.init_process_group(backend=backend, init_method="env://")

    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()
    # port = comm.bcast(_find_free_port(), root=0)
    # os.environ["MASTER_PORT"] = str(port)
    # dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        #return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        return th.device(f"cuda:0")
    return th.device("cpu")
    # if th.cuda.is_available():
    #     return th.device(f"cuda")
    # return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
    # chunk_size = 2 ** 30  # MPI has a relatively small size limit
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    #     num_chunks = len(data) // chunk_size
    #     if len(data) % chunk_size:
    #         num_chunks += 1
    #     MPI.COMM_WORLD.bcast(num_chunks)
    #     for i in range(0, len(data), chunk_size):
    #         MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    # else:
    #     num_chunks = MPI.COMM_WORLD.bcast(None)
    #     data = bytes()
    #     for _ in range(num_chunks):
    #         data += MPI.COMM_WORLD.bcast(None)

    # return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
