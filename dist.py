import os
import sys
import time
import subprocess
import argparse
import torch
import torch.distributed as dist
from torch.autograd import Variable
from utils.generic_utils import load_config, create_experiment_folder


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(rank, num_gpus, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        "nccl",
        init_method="tcp://localhost:54321",
        world_size=num_gpus,
        rank=rank,
        group_name=group_name)


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def apply_gradient_allreduce(module):
    """
    Modifies existing model to do gradient allreduce, but doesn't change class
    so you don't need "module"
    """
    if not hasattr(dist, '_backend'):
        module.warn_on_half = True
    else:
        module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if (module.needs_reduction):
            module.needs_reduction = False
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = type(param.data)
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            if module.warn_on_half:
                if torch.cuda.HalfTensor in buckets:
                    print(
                        "WARNING: gloo dist backend for half parameters may be extremely slow."
                        +
                        " It is recommended to use the NCCL backend in this case. This currently requires"
                        + "PyTorch built from top of tree master.")
                    module.warn_on_half = False

            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(
                        grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):

        def allreduce_hook(*unused):
            Variable._execution_engine.queue_callback(allreduce_params)

        if param.requires_grad:
            param.register_hook(allreduce_hook)
            dir(param)

    def set_needs_reduction(self, input, output):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main(args):
    """
    Call train.py as a new process and pass command arguments
    """
    CONFIG = load_config(args.config_path)
    OUT_PATH = create_experiment_folder(CONFIG.output_path, CONFIG.model_name, False)
    stdout_path = os.path.join(OUT_PATH, "process_stdout/")

    num_gpus = torch.cuda.device_count()
    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    args_list = ['train.py']
    args_list.append('--restore_path={}'.format(args.restore_path))
    args_list.append('--config_path={}'.format(args.config_path))
    args_list.append("--group_id=group_{}".format(group_id))
    args_list.append('--data_path={}'.format(args.data_path))

    if not os.path.isdir(stdout_path):
        os.makedirs(stdout_path)
        os.chmod(stdout_path, 0o775)

    # run processes
    processes = []
    for i in range(num_gpus):
        args_list.append('--rank={}'.format(i))
        stdout = None if i == 0 else open(
            os.path.join(stdout_path, "process_{}.log".format(i)), "w")
        print(args_list)
        p = subprocess.Popen([str(sys.executable)] + args_list, stdout=stdout)
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Folder path to checkpoints',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='path to config file for training',
    )
    parser.add_argument(
        '--data_path', type=str, help='dataset path.', default='')

    args = parser.parse_args()
    main(args)
