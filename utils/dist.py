import os
import torch
import torch.distributed as dist

def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 0:
        rt /= world_size
    return rt
