import sys
import random
import logging

import numpy as np
import torch
import pickle
from datetime import datetime


def set_seed(seed):
    """Sets random seed everywhere."""
    print("Seed set")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm


def get_logger(logger_name = 'marg', level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    if logger_name:
        fh = logging.FileHandler(f'./logs/{logger_name}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log')
        formatter = logging.Formatter(
        fmt="%(asctime)s\n%(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)
