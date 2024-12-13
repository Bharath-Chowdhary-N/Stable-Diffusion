import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention():
    """
    Input = (seq, d_model) # seq     --> sequence length (implies sequence of words / input params)
                           # d_model --> latent representation (typically in paper referred as 512)

    W_input = (d_model, 3*d_model) # W_Input --> input weights for W_Q, W_K, W_V , now represented in one array for better alloc                                                  
    W_output = (d_model, d_model) #W_output --> output weights for projection
    d_k = d_model / h # as represented in paper, for splitting the Q,K,V into smaller tensors
    """
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.W_input = nn.Linear(self.d_model, 3*self.d_model)
        self.W_output = nn.Linear(self.d_model, self.d_model)
        self.n_heads = n_heads
        self.d_k = self.d_model / self.n_heads
        assert isinstance(self.d_k, int)
    def forward(self, x: torch.tensor, causal_mask: bool):
        """
        x: (batch_size, seq, d_model)
        """
        batch_size, seq, d_model = x.shape
        
        interim_dim = (batch_size, seq, self.n_heads, self.d_k) #splitting the input into n_heads

        #Step1: Do the projection with W_input
        Q_dash, K_dash, V_dash = self.W_input(x).chunk(3, dim=-1) #each will have shape (batch_size, seq, d_model)

        # conversion from (batch_size, seq, d_model) --> (batch_size, seq, self.n_heads, self.d_k) --> (batch_size, self.n_heads, seq, self.d_k)
        q_dash = Q_dash.view(interim_dim).transpose(1,2)
        k_dash = K_dash.view(interim_dim).transpose(1,2)
        v_dash = V_dash.view(interim_dim).transpose(1,2)

        qk_product = q_dash @ k_dash.transpose(-1,-2)  # (batch_size, self.n_heads, seq, seq)

        if causal_mask:
            mask = torch.ones_like(qk_product, dtype=bool).triu(1)
            qk_product.masked_fill_(mask, -torch.inf)

        qk_product /= math.sqrt(self.d_k)

        qk_product = F.softmax(qk_product, dim=-1)

        attention_qkv = qk_product @ v_dash 
