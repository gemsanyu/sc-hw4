from typing import Optional, Tuple
import torch
import numpy as np
from torch import nn
import math


# class SkipConnection(nn.Module):
class SkipConnection(torch.jit.ScriptModule):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    @torch.jit.script_method
    def forward(self, input:torch.Tensor):
        return input + self.module(input)


class MultiHeadAttention(torch.jit.ScriptModule):
# class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    
    @torch.jit.script_method
    def forward(self, q:torch.Tensor, h:Optional[torch.Tensor]=None, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        # if mask is not None:
        #     mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        #     compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        # if mask is not None:
        #     attnc = attn.clone()
        #     attnc[mask] = 0
        #     attn = attnc

        heads = torch.matmul(attn, V) #-> weighted average of V, 1 vektor

        # combine the heads by concatenation, and then project the combined by linear layer
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


# class Normalization(nn.Module):
class Normalization(torch.jit.ScriptModule):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()
        # self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True)
        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    @torch.jit.script_method
    def forward(self, input: torch.Tensor)->torch.Tensor:
        #input size = batch_size, num_items, dims
        # batch_size, num_items, feature_size = input.shape
        # input = input.view(batch_size*num_items, feature_size)
        # print(input.shape)
        # normed = self.normalizer(input)
        # normed_ = normed.view(batch_size, num_items, -1)
        return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        # return normed_
        

class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim)
        )


class GraphAttentionEncoder(torch.jit.ScriptModule):
# class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    @torch.jit.script_method
    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:

        # Batch multiply to get initial embeddings of nodes
        h = x
        if self.init_embed is not None:
            h = self.init_embed(x)
        # h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h_ = self.layers(h)

        return (
            h_,  # (1, num_nodes, embed_dim)
            h_.mean(dim=1, keepdim=True),  # average to get embedding of graph, (batch_size, embed_dim)
        )
        