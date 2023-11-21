from functools import partial 
from typing import Optional, Tuple

import tinygrad 
from tinygrad.tensor import Tensor 
from tinygrad.helpers import dtypes

from collections import namedtuple 
from functools import wraps 
from packaging import version 
from dataclasses import dataclass 

from einops import rearrange, repeat 

@dataclass 
class Intermediates:
    qk_similarities: Optional[Tensor] = None 
    pre_softmax_attn: Optional[Tensor] = None 
    post_softmax_attn: Optional[Tensor] = None 
    cached_kv: Optional[Tuple[Tensor, Tensor]] = None 

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None 

def default(val, d):
    return val if exists(val) else d 

def compact(arr):
    return [*filter(exists, arr)]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called 
        if called:
            return 
        called = True 
        return fn(x)
    return inner 

print_once = once(print)

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return Tensor.ones(i, j, device=device, dtype=dtypes.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = Tensor.arange(i, device=device)
    causal_mask = rearrange(r.numpy(), 'i -> i 1') < rearrange(r.numpy(), 'j -> 1 j')
    causal_mask = np.pad(causal_mask, pad_width=(j - i, 0), mode='constant', constant_values=0)
    return Tensor(causal_mask)

# main class 

class Attend():
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        talking_heads = False,
        sparse_topk = None,
        scale = None,
        qk_norm = False,
        flash = False,
        add_zero_kv = False,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        self.scale = scale 
        self.qk_norm = qk_norm 
        self.causal = causal 
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask 

        self.dropout = dropout 
        self.attn_dropout = 
