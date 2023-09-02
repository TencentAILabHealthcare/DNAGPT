# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/3/5
import os
import importlib
import sys
from functools import partial
import inspect

__all__ = ["get_registry", "Registry"]

GLOBAL_REGISTRY = {}


def auto_import(moudle_dir="."):
    for file in os.listdir(moudle_dir):
        path = os.path.join(moudle_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module("." + model_name)


def _register_generic(module_dict, module_name, module):
    if module_name in module_dict:
        pass
    else:
        module_dict[module_name] = module


def get_registry(name):
    reg = GLOBAL_REGISTRY.get(name, None)
    if reg is None:
        print(f"not registry named: {name}", file=sys.stderr)

    return reg


class Registry(dict):
    """
    registry helper，提供注册接口
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    两种方式注册新的模块:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)

    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    通过字典获取注册模块, eg:
        f = some_registry["foo_modeul"]
    """

    def __init__(self, name="registry", *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self.name = name
        GLOBAL_REGISTRY[name] = self

    @classmethod
    def get_registry(cls, name):
        reg = GLOBAL_REGISTRY.get(name, None)
        if reg is None:
            print(f"not registry named: {name}", file=sys.stderr)

        return reg

    def register(self, module_name=None, module=None, *args, **kwargs):
        """注册模块, 这里作为装饰器
        Args:
            module_name: str, 模块名
            module： function
        """
        # 作为方程使用
        if module is not None:
            if len(kwargs):
                module = partial(module, *args, **kwargs)
            _register_generic(self, module_name, module)

            return

        # 作为装饰器使用
        def register_fn(fn):
            name = module_name if module_name else fn.__name__
            if len(kwargs):
                fn = partial(fn, *args, **kwargs)
            _register_generic(self, name, fn)

            return fn

        return register_fn


def build_from_cfg(cfg, registry, **kwargs):
    """Build a module from config dict.

    Args:
        cfg: Config dict. It should at least contain the key "type".
        registry: The registry to search the type from.
        default_args: Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')

    args = cfg.copy()
    obj_type = args["type"]

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    for name, value in kwargs.items():
        if name in args:
            print(f"override param: {name}, old: {args[name]} new； {value}")
        args[name] = value

    return obj_cls(**args)
