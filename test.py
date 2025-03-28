import torch
import numpy as np
import transformers
import lightning
import PIL
import tqdm

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
