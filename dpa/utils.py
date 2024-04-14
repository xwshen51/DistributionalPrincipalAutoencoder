import os
import torch

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def onehot2num(a):
    return torch.argmax(a, dim=1)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")
