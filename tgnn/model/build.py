# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/3/30
from functools import partial
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import fsdp
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tgnn.utils.registry import Registry
from tgnn.utils import comm, get_torch_version

MODEL_REGISTRY = Registry("MODEL_ARCH")  # noqa F401 isort:skip
MODEL_REGISTRY.__doc__ = """
Registry for meta-architectures,

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model

    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]

    ddp = DistributedDataParallel(model, **kwargs)

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    return ddp


def warp_fsdp_model(cfg, model):
    auto_wrap_policy = None
    if cfg.MODEL.FSDP.POLICY == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=cfg.MODEL.FSDP.MIN_PARAMS)
    elif cfg.MODEL.FSDP.POLICY == "transformer":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=(model.Block, ))

    cpu_offload = CPUOffload(offload_params=True) if cfg.MODEL.FSDP.CPU_OFFLOAD else None
    assert cfg.SOLVER.DTYPE in ("float16", "bfloat16", "float32"), f"only support float type, get{cfg.SOLVER.DTYPE}"
    dtype = getattr(torch, cfg.SOLVER.DTYPE)
    mixed_precision = fsdp.MixedPrecision(param_dtype=dtype,
                                          # Gradient communication precision.
                                          reduce_dtype=dtype,
                                          # Buffer precision.
                                          buffer_dtype=dtype)
    backward_prefetch = getattr(BackwardPrefetch, f"BACKWARD_{cfg.MODEL.FSDP.BACKWARD_PREFETCH.upper()}")
    sharding_strategy = getattr(fsdp.ShardingStrategy, cfg.MODEL.FSDP.SHARDING_STRATEGY)
    model = FSDP(model,
                 auto_wrap_policy=auto_wrap_policy,
                 cpu_offload=cpu_offload,
                 mixed_precision=mixed_precision,
                 backward_prefetch=backward_prefetch,
                 sharding_strategy=sharding_strategy)

    return model


def build_model(cfg):
    """build model by architecture name
    Use single-machine multi-GPU DataParallel(dp),
    you would like to speed up training with the minimum code change.
    for fast traning speed, recommend using ddp model
    """
    arch = cfg.MODEL.ARCH

    assert arch in MODEL_REGISTRY, f"{arch} not in model registry"

    model = MODEL_REGISTRY.get(arch)(cfg)
    torch._C._log_api_usage_once(f"tgnn.model.meta_arch.{arch}")

    if cfg.MODEL.COMPILE:
        assert get_torch_version() >= [2, 0], f"only pytorch 2.0 support model compile, get {torch.__version__}"
        model = torch.compile(model)  # requires PyTorch 2.0

    if cfg.CUDA:
        model.cuda()
        if comm.get_world_size() > 1:
            if cfg.MODEL.FSDP.ENABLED:
                model = warp_fsdp_model(cfg, model)
            else:
                model = create_ddp_model(model,
                                         broadcast_buffers=cfg.MODEL.DDP.BROADCAST_BUFFERS)

            if cfg.MODEL.SYN_BN:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
