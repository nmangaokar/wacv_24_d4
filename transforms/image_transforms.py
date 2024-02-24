import torch_dct as dct
import torch

def holz_transform(tensor):
    tensor = dct.dct_2d(tensor)
    tensor = torch.abs(tensor)
    tensor += 1e-12
    tensor = torch.log(tensor)
    return tensor

