import os
from os.path import join as pjoin
import time
import logging
from importlib.machinery import SourceFileLoader

import yaml
import torch
from attrdict import AttrDict

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()

def load_task(task_name):
    task_cls = getattr(load_module(f'task/{task_name}.py'), task_name.upper())
    with open(f'config/{task_name}/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    task = task_cls(**config)
    return task

def load_model(task_name, model_name):
    model_cls = getattr(load_module(f'models/{model_name}.py'), model_name.upper())
    with open(f'config/{task_name}/{model_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()
    return model

def load_results_path(task_name, model_name, exp_id):
    results_path = pjoin('./results', task_name, model_name, exp_id)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    return results_path

def load_logger(results_path, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(pjoin(results_path, f'train_{time.strftime("%Y%m%d-%H%M")}.log'), mode=mode))
    return logger

def save_info(args):
    with open(pjoin(f'{args.results_path}/args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

def save_model(results_path, model, optimizer, scheduler, step):
    ckpt = AttrDict()
    ckpt.model = model.state_dict()
    ckpt.optimizer = optimizer.state_dict()
    ckpt.scheduler = scheduler.state_dict()
    ckpt.step = step
    torch.save(ckpt, pjoin(results_path, 'ckpt.tar'))