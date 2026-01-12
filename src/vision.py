import math 
import torch 
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass 
class VisionConfig:
    