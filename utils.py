import random
import numpy as np
import torch


def none_or_str(value: str):
    if value.lower() == 'none':
        return None
    return value


def str_to_bool(value: str):
    if value.lower() == 'yes' or value.lower() == 'true':
        return True
    elif value.lower() == 'no' or value.lower() == 'false':
        return False
    else:
        raise ValueError


def set_seed(seed=0):
    """Set the random seed for torch.
    Args:
        seed (int, optional): random seed. Default 0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is not available, this is silently ignored.
    torch.cuda.manual_seed_all(seed)


def load_text_file(file_path):
    """
    Load text file into a list, each item is a line of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def write_text_file(lines, file_path):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")