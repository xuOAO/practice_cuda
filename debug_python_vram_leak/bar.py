import torch

torch.cuda.set_device(0)
my_stream = torch.cuda.Stream()