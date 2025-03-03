
import torch
from torch import nn
import lavin
from typing import Optional, Tuple
from  torch.cuda.amp import autocast
import lavin.eval_model


class RepAdapter_Router(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1,
            t=10.
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights=nn.Linear(in_features,2)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale
        self.t=t

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)


        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

    def forward(self, x,weights=None):
        with autocast():
            if weights is None:
                weights=torch.softmax(self.expert_weights(x[:,0])/self.t,-1).half()
            x=x.transpose(1,2)
            x_=self.dropout(self.conv_A(x))
            x=self.conv_B(x_)*self.scale*weights[:,0,None,None]+self.conv_D(x_)*self.scale*weights[:,1,None,None]+x
            x=x.transpose(1,2).contiguous()
        return x



class RepAdapter(nn.Module):
    """
    Pytorch Implemention of RepAdapter for 1d tensor
    copy from https://github.com/luogen1996/RepAdapter/blob/main/repadapter.py
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x,weights=None):
        res = x
        with autocast():
            x=x.transpose(1,2)
            x=self.conv_B(self.dropout(self.conv_A(x)))
            x=x.transpose(1,2).contiguous()
        return (res + x).float()


def forward_llama_block(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.adapter_mlp(self.ffn_norm(h))))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h))))
    return out

def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out
def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_=x.shape[0]
    if start_pos==0:
        self.cache_weights[:bs_]=torch.softmax(self.adapter_attn.expert_weights(self.attention_norm(x)[:,0].float())/self.t,-1).half()
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x),weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out

def forward_llama_block_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_=x.shape[0]
    if start_pos==0:
        self.cache_weights[:bs_]=torch.softmax(self.adapter_attn.expert_weights(self.attention_norm(x)[:,0].float())/self.t,-1).half()
        self.cache_weights_ffn[:bs_]=torch.softmax(self.adapter_mlp.expert_weights(self.ffn_norm(x)[:,0].float())/self.t,-1).half()
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x),weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h),self.cache_weights_ffn[:bs_])))
    return out


def forward_vivit(self, hidden_states, head_mask=None, output_attentions=False):
    self_attention_outputs = self.attention(
        # in Vivit, layernorm is applied before self-attention
        self.adapter_attn(self.layernorm_before(hidden_states)),
        head_mask,
        output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]
    # add self attentions if we output attention weights
    outputs = self_attention_outputs[1:]

    # first residual connection
    hidden_states = attention_output + hidden_states

    # in Vivit, layernorm is also applied after self-attention
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.intermediate(layer_output)

    # second residual connection is done here
    layer_output = self.output(layer_output, hidden_states)

    outputs = (layer_output,) + outputs

    return outputs


def forward_vivit_full(self, hidden_states, head_mask=None, output_attentions=False):
    self_attention_outputs = self.attention(
        # in Vivit, layernorm is applied before self-attention
        self.adapter_attn(self.layernorm_before(hidden_states)),
        head_mask,
        output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]
    # add self attentions if we output attention weights
    outputs = self_attention_outputs[1:]

    # first residual connection
    hidden_states = attention_output + hidden_states

    # in Vivit, layernorm is also applied after self-attention
    layer_output = self.adapter_mlp(self.layernorm_after(hidden_states))
    layer_output = self.intermediate(layer_output)

    # second residual connection is done here
    layer_output = self.output(layer_output, hidden_states)

    outputs = (layer_output,) + outputs

    return outputs


def forward_whisper(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.adapter_attn(self.self_attn_layer_norm(hidden_states))
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs

def forward_whisper_full(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.adapter_attn(self.self_attn_layer_norm(hidden_states))
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.adapter_mlp(self.final_layer_norm(hidden_states))
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def set_MMAdapter(model, method, dim=8, s=1, set_forward=True,t=10,gradient_checkpointing=False):
    if method == 'block':
        # not support right now
        assert NotImplementedError
        for _ in model.children():
            if type(_) ==  lavin.model.TransformerBlock or type(_) == lavin.eval_model.TransformerBlock:
                _.adapter_attn = RepAdapter_Router(_.dim,hidden_dim=dim,scale=s,t=t)
                _.adapter_mlp = RepAdapter_Router(_.dim,hidden_dim=dim,scale=s,t=t)
                _.s = s
                _.t = t
                _.gradient_checkpointing=gradient_checkpointing
                if type(_) == lavin.eval_model.TransformerBlock:
                    bound_method = forward_llama_block_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_block.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_MMAdapter(_, method, dim, s,set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)

    else:
        for _ in model.children():
            if type(_) == lavin.model.TransformerBlock or type(_) == lavin.eval_model.TransformerBlock:
                _.adapter_attn = RepAdapter_Router(_.dim,hidden_dim=dim,scale=s,t=t)
                _.s = s
                _.t=t
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == lavin.eval_model.TransformerBlock:
                    bound_method = forward_llama_attn_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_attn.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_MMAdapter(_, method, dim, s, set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)


from vivit import VivitLayer
def set_Vivit_Adapter(model, method, dim=8, s=1, set_forward=True, t=10.):
    for _ in model.children():
        if type(_) == VivitLayer:
            if method=='router':
                _.adapter_attn = RepAdapter_Router(768, hidden_dim=dim, scale=s,  t=t)
            elif method=='router_block':
                _.adapter_attn = RepAdapter_Router(768, hidden_dim=dim, scale=s,  t=t)
                _.adapter_mlp = RepAdapter_Router(768, hidden_dim=dim, scale=s,  t=t)
            else:
                _.adapter_attn = RepAdapter(768, hidden_dim=dim, scale=s)
            _.s = s
            if method=='router_block':
                bound_method = forward_vivit_full.__get__(_, _.__class__)
            else:
                bound_method = forward_vivit.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Vivit_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
    return

from whisper.modeling_whisper import WhisperEncoderLayer
def set_Whisper_Adapter(model, method, dim=8, s=1, set_forward=True, t=10.):
    for _ in model.children():
        if type(_) == WhisperEncoderLayer:
            if method=='router':
                _.adapter_attn = RepAdapter_Router(1280, hidden_dim=dim, scale=s,  t=t)
            elif method=='router_block':
                _.adapter_attn = RepAdapter_Router(1280, hidden_dim=dim, scale=s,  t=t)
                _.adapter_mlp = RepAdapter_Router(1280, hidden_dim=dim, scale=s,  t=t)
            else:
                _.adapter_attn = RepAdapter(1280, hidden_dim=dim, scale=s)
            _.s = s
            if method=='router_block':
                bound_method = forward_whisper_full.__get__(_, _.__class__)
            else:
                bound_method = forward_whisper.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Whisper_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
    return