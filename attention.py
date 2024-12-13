import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention():
    """
    Input = (seq, d_model) # seq     --> sequence length (implies sequence of words / input params)
                           # d_model --> latent representation (typically in paper referred as 512)

    W_Input = (d_model, 3*d_model) # W_Input --> input weights for W_Q, W_K, W_V , now represented in one array for better alloc                                                  
    W_Output = (d_model, d_model) #W_output --> output weights for projection
    """
    def __init__(self):
        pass
