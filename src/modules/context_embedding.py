from torch import nn
from torch.nn import functional as F
from .utils import *
from .helpers import *
from einops import rearrange
from torch import einsum
from typing import Union, Dict


class CrossAttention(nn.Module):
    # Ref:Section 3.3 from High-Resolution Image Synthesis with Latent Diffusion Models
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def _attention(self, q, k, v, h, mask, scale, is_style):
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * scale

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return out

    def forward(self, x, context=None, mask=None, is_style=False):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        out = self._attention(q, k, v, h, mask, self.scale, is_style)

        return self.to_out(out)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    #'seq shape', seq.shape)
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention for the image
        self.attnc = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, trigger_style_attn=True, is_style=False):
        if self.training or not trigger_style_attn:
            # print('go here')
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            # print('go there')
            return checkpoint(
                self._forward,
                (x, context, is_style),
                self.parameters(),
                self.checkpoint,
            )

    def _forward(self, x, context=None, is_style=False):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=None, is_style=is_style) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        part="encoder",
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head  # inner is equal to self.in_cahnnels
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        self.part = part

    def forward(
        self,
        x,
        context: Union[Dict[str, torch.Tensor], torch.Tensor],
        trigger_style_attn=True,
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        # note: if no context is given, cross-attention defaults to self-attention

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        if self.part != "sca":
            x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.transformer_blocks:
            if isinstance(context, Dict):
                x = block(
                    x,
                    context=context["style"],
                    trigger_style_attn=trigger_style_attn,
                    is_style=True,
                )
                x = block(x, context=context["text"], trigger_style_attn=trigger_style_attn)
            else:
                x = block(x, context=context)

        if self.part != "sca":
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = self.proj_out(x)

        return x + x_in


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, trigger_style_attn=True):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, trigger_style_attn)
            else:
                x = layer(x)

        return x
