# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    VocabParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

from torch.nn import Embedding, Linear
import torch
import pdb
from timm.models.layers import  DropPath
import clip
import vivit
import whisper
from  torch.cuda.amp import autocast
# @dataclass
# class ModelArgs:
#     dim: int = 512
#     n_layers: int = 8
#     n_heads: int = 8
#     vocab_size: int = -1  # defined later by tokenizer
#     multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
#     norm_eps: float = 1e-5
#     hidden_proj: int=128

#     max_batch_size: int = 32
#     max_seq_len: int = 2048
#     drop_path: float=0.


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float=0.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        #modified bias for reparameterizing
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            self.n_local_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            self.n_local_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        # ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         model_parallel_size = fs_init.get_model_parallel_world_size()
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.wq = ColumnParallelLinear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wk = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wv = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wo = RowParallelLinear(
#             args.n_heads * self.head_dim,
#             args.dim,
#             bias=False,
#             input_is_parallel=True,
#             init_method=lambda x: x,
#         )

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         bsz, seqlen, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         self.cache_k = self.cache_k.to(xq)
#         self.cache_v = self.cache_v.to(xq)

#         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         keys = self.cache_k[:bsz, : start_pos + seqlen]
#         values = self.cache_v[:bsz, : start_pos + seqlen]

#         # repeat k/v heads if n_kv_heads < n_heads
#         keys = repeat_kv(
#             keys, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
#         values = repeat_kv(
#             values, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         values = values.transpose(
#             1, 2
#         )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)


# class FeedForward(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: int,
#         multiple_of: int,
#     ):
#         super().__init__()
#         hidden_dim = int(2 * hidden_dim / 3)
#         hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

#         self.w1 = Linear(
#             dim, hidden_dim, bias=False
#         )
#         self.w2 = Linear(
#             hidden_dim, dim, bias=False
#         )
#         self.w3 = Linear(
#             dim, hidden_dim, bias=False
#         )

#     def forward(self, x):
#         return self.w2(F.silu(self.w1(x),inplace=False) * self.w3(x))

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# class TransformerBlock(nn.Module):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Attention(args)
#         self.feed_forward = FeedForward(
#             dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
#         )
#         self.layer_id = layer_id
#         self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

#     def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

#         h = x + self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter))
#         out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
#         return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.drop_path(self.attention(self.attention_norm(x), start_pos, freqs_cis, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out



class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.conv_A=nn.Linear(in_features,hidden_dim)
        self.conv_B = nn.Linear(hidden_dim, out_features)


        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        with autocast():
            x=self.conv_B(F.silu(self.conv_A(x)))
        return x


# class Transformer(nn.Module):
#     def __init__(self, params: ModelArgs):
#         super().__init__()
#         self.params = params
#         self.vocab_size = params.vocab_size
#         self.n_layers = params.n_layers
#         self.tok_embeddings = Embedding(
#             params.vocab_size, params.dim
#         )


#         self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

#         # with init_empty_weights():
#         self.layers = torch.nn.ModuleList()
#         for layer_id in range(params.n_layers):
#             self.layers.append(TransformerBlock(layer_id, params))

#         self.norm = RMSNorm(params.dim, eps=params.norm_eps)
#         self.output = Linear(
#             params.dim, params.vocab_size, bias=False
#         )

#         self.freqs_cis = precompute_freqs_cis(
#             self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
#         )

#         self.backbone = clip.load('ViT-L/14')[0]


#         #handcraft define self.backbone.visual.transformer.width
#         self.adapter_proj = AdapterMLP(1024, params.hidden_proj, params.dim).float()
#         self.adapter_modality_embedding=nn.Embedding(2,params.dim).float()



#     def insert_image_embeds(self,examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators):
#         _bsz, seqlen,_ = examples.shape
#         new_examples=[]
#         new_labels=[]
#         for i, (example,label) in enumerate(zip(examples,labels)):
#             if img_indicators[i]>0.:
#                 new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
#                 new_label=torch.cat([label[:1],
#                                      torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
#                                      label[1:]])
#                 new_example = new_example[:seqlen]
#                 new_label = new_label[:seqlen]
#             else:
#                 new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
#                 new_label=torch.cat([label[:1],
#                                      torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
#                                      label[1:]])
#                 new_example = new_example[:seqlen]
#                 new_label = new_label[:seqlen]
#             new_examples.append(new_example.unsqueeze(0))
#             new_labels.append(new_label.unsqueeze(0))
#         new_examples = torch.cat(new_examples, 0)
#         new_labels = torch.cat(new_labels, 0)
#         return new_examples,new_labels

#     def forward(self, examples, labels,images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None):

#         # print(images.dtype)
#         examples = examples.cuda()
#         labels = labels.cuda()
#         images = images.cuda()
#         prefix_img = prefix_img.cuda()
#         prefix_nonimg = prefix_nonimg.cuda()
#         image_embeds = self.backbone.encode_image(images).half()

#         # print(img_indicators)
#         if isinstance(img_indicators,list):
#             img_indicators = torch.Tensor(img_indicators).long()
#         img_indicators = img_indicators.to(image_embeds.device)
#         modality_embed=self.adapter_modality_embedding(img_indicators.unsqueeze(1))

#         # with autocast():
#         image_embeds=self.adapter_proj(image_embeds)

#         # print(image_embeds.shape)

#         _bsz, seqlen = examples.shape

#         examples = self.tok_embeddings(examples)
#         prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)
#         prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0)


#         h,labels=self.insert_image_embeds(examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators)

#         h=torch.cat([modality_embed.half(),h],1)[:,:seqlen]
#         modality_labels=torch.zeros(_bsz,1).to(labels.device).type_as(labels)
#         labels=torch.cat([modality_labels,labels],1)[:,:seqlen]


#         freqs_cis = self.freqs_cis.to(h.device)
#         freqs_cis = freqs_cis[:seqlen]
#         mask = None
#         mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
#         mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

#         #mask decision token
#         mask[:,:,1:,0]=float("-inf")

#         start_pos = 0
#         for layer in self.layers:
#             h = layer(h, start_pos, freqs_cis, mask)

#         h = self.norm(h)
#         output = self.output(h)
#         output = output[:, :-1, :].reshape(-1, self.vocab_size)
#         labels = labels[:, 1:].flatten()


#         c_loss = self.criterion(output, labels)
#         return c_loss

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )
        # self.tok_embeddings = VocabParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        # self.backbone = clip.load('ViT-L/14')[0]
        self.video_backbone = vivit.VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.audio_backbone = whisper.WhisperModel.from_pretrained("openai/whisper-large-v3").encoder

        #handcraft define self.backbone.visual.transformer.width
        self.video_adapter_proj = AdapterMLP(768, params.hidden_proj, params.dim).float()
        self.audio_adapter_proj = AdapterMLP(1280, params.hidden_proj, params.dim).float()
        self.adapter_modality_embedding=nn.Embedding(2,params.dim).float()


    def insert_image_embeds(self,examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators):
        _bsz, seqlen,_ = examples.shape
        new_examples=[]
        new_labels=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            if img_indicators[i]>0.:
                new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
                new_label=torch.cat([label[:1],
                                     torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                                     label[1:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            else:
                new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
                new_label=torch.cat([label[:1],
                                     torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
                                     label[1:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)
        return new_examples,new_labels

    def insert_audio_video_embeds(self, examples, labels, video_embeds, prefix_video, audio_embeds, prefix_audio):
        _bsz, seqlen,_ = examples.shape
        new_examples=[]
        new_labels=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            new_example=torch.cat([example[:1],prefix_video,video_embeds[i],prefix_audio,audio_embeds[i],example[1:]],0)
            new_label=torch.cat([label[:1],
                                    torch.zeros(prefix_video.shape[0]+video_embeds.shape[1]+prefix_audio.shape[0]+audio_embeds.shape[1]).to(examples.device).type_as(labels),
                                    label[1:]])
            new_example = new_example[:seqlen]
            new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)    
        return new_examples, new_labels

    # def forward(self, examples, labels,images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None):

    #     # print(images.dtype)
    #     examples = examples.cuda()
    #     labels = labels.cuda()
    #     images = images.cuda()
    #     prefix_img = prefix_img.cuda()
    #     prefix_nonimg = prefix_nonimg.cuda()
    #     image_embeds = self.backbone.encode_image(images).half()

    #     # print(img_indicators)
    #     if isinstance(img_indicators,list):
    #         img_indicators = torch.Tensor(img_indicators).long()
    #     img_indicators = img_indicators.to(image_embeds.device)
    #     modality_embed=self.adapter_modality_embedding(img_indicators.unsqueeze(1))

    #     # with autocast():
    #     image_embeds=self.adapter_proj(image_embeds)

    #     # print(image_embeds.shape)

    #     _bsz, seqlen = examples.shape

    #     examples = self.tok_embeddings(examples)
    #     prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)
    #     prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0)


    #     h,labels=self.insert_image_embeds(examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators)

    #     h=torch.cat([modality_embed.half(),h],1)[:,:seqlen]
    #     modality_labels=torch.zeros(_bsz,1).to(labels.device).type_as(labels)
    #     labels=torch.cat([modality_labels,labels],1)[:,:seqlen]


    #     freqs_cis = self.freqs_cis.to(h.device)
    #     freqs_cis = freqs_cis[:seqlen]
    #     mask = None
    #     mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    #     mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

    #     #mask decision token
    #     mask[:,:,1:,0]=float("-inf")

    #     start_pos = 0
    #     for layer in self.layers:
    #         h = layer(h, start_pos, freqs_cis, mask)

    #     h = self.norm(h)
    #     output = self.output(h)
    #     output = output[:, :-1, :].reshape(-1, self.vocab_size)
    #     labels = labels[:, 1:].flatten()


    #     c_loss = self.criterion(output, labels)
    #     return c_loss
    
    def forward(self, examples, labels, videos = None, prefix_video = None, audios = None, prefix_audio = None, return_hidden=False):

        # print(images.dtype)
        examples = examples.cuda()
        labels = labels.cuda()
        videos = videos.cuda()
        audios =  audios.cuda()
        prefix_video = prefix_video.cuda()
        prefix_audio = prefix_audio.cuda()
        video_embeds = self.video_backbone(videos.half())
        audio_embeds = self.audio_backbone(audios.half())

        # with autocast():
        video_embeds=self.video_adapter_proj(video_embeds)
        audio_embeds=self.audio_adapter_proj(audio_embeds)

        # print(image_embeds.shape)

        _bsz, seqlen = examples.shape

        modality_embed=self.adapter_modality_embedding(torch.ones(_bsz,1).long().cuda())

        examples = self.tok_embeddings(examples)
        prefix_audio=self.tok_embeddings(prefix_audio.unsqueeze(0)).squeeze(0)
        prefix_video=self.tok_embeddings(prefix_video.unsqueeze(0)).squeeze(0)


        h,labels=self.insert_audio_video_embeds(examples, labels, video_embeds, prefix_video, audio_embeds, prefix_audio)

        h=torch.cat([modality_embed.half(),h],1)[:,:seqlen]
        modality_labels=torch.zeros(_bsz,1).to(labels.device).type_as(labels)
        labels=torch.cat([modality_labels,labels],1)[:,:seqlen]


        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        #mask decision token
        mask[:,:,1:,0]=float("-inf")

        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        if return_hidden:
            non_zero_mask = labels != 0
            max_indices = torch.zeros(labels.size(0), dtype=torch.long) - 1
            for i in range(labels.size(0)):
                if non_zero_mask[i].any():
                    max_indices[i] = torch.max(torch.nonzero(non_zero_mask[i])).item()
            return h, max_indices
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()


        c_loss = self.criterion(output, labels)
        return c_loss
    
class TransformerForClassification(nn.Module):
    def __init__(self, params: ModelArgs, num_classes):
        super().__init__()
        self.transformer = Transformer(params)

        self.output_classification = nn.Linear(params.dim, num_classes)
        nn.init.xavier_uniform_( self.output_classification.weight)
        nn.init.zeros_(self.output_classification.bias)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, examples, labels, classes, videos = None, prefix_video = None, audios = None, prefix_audio = None):
        classes = classes.cuda()
        h, max_indices = self.transformer(examples, labels, videos=videos, prefix_video=prefix_video, audios=audios, prefix_audio=prefix_audio, return_hidden=True)
        output = self.output_classification(h[torch.arange(h.shape[0]), max_indices].float())
        loss = self.criterion(output, classes)
        return loss
    
class TransformerForRegression(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.transformer = Transformer(params)

        self.output_regression = nn.Linear(params.dim, 1)
        nn.init.xavier_uniform_( self.output_regression.weight)
        nn.init.zeros_(self.output_regression.bias)
        self.criterion = nn.MSELoss()

    def forward(self, examples, labels, values, videos = None, prefix_video = None, audios = None, prefix_audio = None):
        values = values.cuda()
        h, max_indices = self.transformer(examples, labels, videos=videos, prefix_video=prefix_video, audios=audios, prefix_audio=prefix_audio, return_hidden=True)
        output = self.output_regression(h[torch.arange(h.shape[0]), max_indices].float()).squeeze(1)
        loss = self.criterion(output, values)
        return loss