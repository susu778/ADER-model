import csv
import json
import os
import random
import shutil

import numpy as np
import torch

from ader.entities import TokenSpan

CSV_DELIMITER = ';'


def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMITER, quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMITER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMITER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)


def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMITER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)

    return lines[0], lines[1:]


def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith('/') else source + '/'
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = '/'.join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)

        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue

        create_directories_dir(new_dir)

        for file_name in file_names:
            if file_name.endswith('.py'):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vars(dic), f, ensure_ascii=False, indent=2)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'

    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy, and PyTorch to improve reproducibility.
    """
    if seed is None:
        return

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    torch.use_deterministic_algorithms(True, warn_only=True)


    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int):

    worker_seed = torch.initial_seed() % (2 ** 32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilter(f)


def flatten(l):
    return [i for p in l for i in p]


def get_as_list(dic, key):
    if key in dic:
        return [dic[key]]
    else:
        return []


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_nonzero(tensor, padding=0):
    indices = padded_stack([tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding)
    return indices


def swap(v1, v2):
    return v2, v1


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        if t.span[0] == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.span[1] == span[1]:
            return TokenSpan(span_tokens)

    return None


def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


def load_state_dict(model_path):
    """
    Load a PyTorch or safetensors state dictionary from a model path.
    """
    if os.path.isdir(model_path):
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(model_path, "model.safetensors")

        if os.path.exists(safetensors_path):
            target_path = safetensors_path
            is_safetensors = True
        elif os.path.exists(bin_path):
            target_path = bin_path
            is_safetensors = False
        else:
            raise FileNotFoundError(
                f"Neither pytorch_model.bin nor model.safetensors was found in {model_path}"
            )
    else:
        target_path = model_path
        is_safetensors = model_path.endswith(".safetensors")

    print(f"Loading model weights from: {target_path}")

    if is_safetensors:
        if load_safetensors is None:
            raise ImportError(
                "A .safetensors file was detected, but the safetensors package is not installed. "
                "Please install it with: pip install safetensors"
            )
        return load_safetensors(target_path, device="cpu")

    return torch.load(target_path, map_location=torch.device("cpu"))


def check_version(config, model_class, model_path):
    """
    Check ADER model version and load the corresponding state dictionary.
    """
    state_dict = load_state_dict(model_path)

    config_dict = config.to_dict()
    loaded_version = config_dict.get("ader_version", config_dict.get("spert_version", "1.0"))

    if "rel_classifier.weight" in state_dict and loaded_version != model_class.VERSION:
        msg = (
            f"Current ADER version ({model_class.VERSION}) does not match "
            f"the version of the loaded model ({loaded_version}). "
            "Use the matching code version or train a new model."
        )
        raise Exception(msg)

    return state_dict