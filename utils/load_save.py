import os
from os.path import join as pjoin
from importlib.machinery import SourceFileLoader

import yaml

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()

def load_model(task_name, model_name):
    model_cls = getattr(load_module(f'models/{model_name}.py'), model_name.upper())
    with open(f'config/{task_name}/{model_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()
    return model

def load_results_path(task_name, model_name, exp_id):
    results_path = pjoin(f"./results/{task_name}/{model_name}/{exp_id}/")
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    return results_path