# import math
# from collections import OrderedDict
# from dataclasses import dataclass, replace, field
# from functools import partial
# from typing import Callable, Optional, Union, Tuple, List

# import torch
# from torch import nn
# from torch.jit import Final

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.layers import Mlp, ConvMlp, DropPath, LayerNorm, ClassifierHead, NormMlpClassifierHead
# from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
# from timm.layers import trunc_normal_tf_, to_2tuple, extend_tuple, make_divisible, _assert
# from timm.layers import RelPosMlp, RelPosBias, RelPosBiasTf, use_fused_attn, resize_rel_pos_bias_table
# from timm.models._builder import build_model_with_cfg
# from timm.models._features_fx import register_notrace_function
# from timm.models._manipulate import named_apply, checkpoint_seq
# from timm.models._registry import generate_default_cfgs, register_model


# @dataclass
# class MaxxVitTransformerCfg:
#     dim_head: int = 32
#     head_first: bool = True  # head ordering in qkv channel dim
#     expand_ratio: float = 4.0
#     expand_first: bool = True
#     shortcut_bias: bool = True
#     attn_bias: bool = True
#     attn_drop: float = 0.
#     proj_drop: float = 0.
#     pool_type: str = 'avg2'
#     rel_pos_type: str = 'bias'
#     rel_pos_dim: int = 512  # for relative position types w/ MLP
#     partition_ratio: int = 32
#     window_size: Optional[Tuple[int, int]] = None
#     grid_size: Optional[Tuple[int, int]] = None
#     no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
#     use_nchw_attn: bool = False  # for MaxViT variants (not used for CoAt), keep tensors in NCHW order
#     init_values: Optional[float] = None
#     act_layer: str = 'gelu'
#     norm_layer: str = 'layernorm2d'
#     norm_layer_cl: str = 'layernorm'
#     norm_eps: float = 1e-6

#     def __post_init__(self):
#         if self.grid_size is not None:
#             self.grid_size = to_2tuple(self.grid_size)
#         if self.window_size is not None:
#             self.window_size = to_2tuple(self.window_size)
#             if self.grid_size is None:
#                 self.grid_size = self.window_size


# @dataclass
# class MaxxVitConvCfg:
#     block_type: str = 'mbconv'
#     expand_ratio: float = 4.0
#     expand_output: bool = True  # calculate expansion channels from output (vs input chs)
#     kernel_size: int = 3
#     group_size: int = 1  # 1 == depthwise
#     pre_norm_act: bool = False  # activation after pre-norm
#     output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
#     stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
#     pool_type: str = 'avg2'
#     downsample_pool_type: str = 'avg2'
#     padding: str = ''
#     attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
#     attn_layer: str = 'se'
#     attn_act_layer: str = 'silu'
#     attn_ratio: float = 0.25
#     init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
#     act_layer: str = 'gelu'
#     norm_layer: str = ''
#     norm_layer_cl: str = ''
#     norm_eps: Optional[float] = None

#     def __post_init__(self):
#         # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
#         assert self.block_type in ('mbconv', 'convnext')
#         use_mbconv = self.block_type == 'mbconv'
#         if not self.norm_layer:
#             self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
#         if not self.norm_layer_cl and not use_mbconv:
#             self.norm_layer_cl = 'layernorm'
#         if self.norm_eps is None:
#             self.norm_eps = 1e-5 if use_mbconv else 1e-6
#         self.downsample_pool_type = self.downsample_pool_type or self.pool_type


# @dataclass
# class MaxxVitCfg:
#     embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
#     depths: Tuple[int, ...] = (2, 3, 5, 2)
#     block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
#     stem_width: Union[int, Tuple[int, int]] = 64
#     stem_bias: bool = False
#     conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)
#     transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)
#     head_hidden_size: int = None
#     weight_init: str = 'vit_eff'


# class Attention2d(nn.Module):
#     fused_attn: Final[bool]

#     """ multi-head attention for 2D NCHW tensors"""
#     def __init__(
#             self,
#             dim: int,
#             dim_out: Optional[int] = None,
#             dim_head: int = 32,
#             bias: bool = True,
#             expand_first: bool = True,
#             head_first: bool = True,
#             rel_pos_cls: Callable = None,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.
#     ):
#         super().__init__()
#         dim_out = dim_out or dim
#         dim_attn = dim_out if expand_first else dim
#         self.num_heads = dim_attn // dim_head
#         self.dim_head = dim_head
#         self.head_first = head_first
#         self.scale = dim_head ** -0.5
#         self.fused_attn = use_fused_attn()

#         self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
#         self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
#         B, C, H, W = x.shape

#         if self.head_first:
#             q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
#         else:
#             q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

#         if self.fused_attn:
#             attn_bias = None
#             if self.rel_pos is not None:
#                 attn_bias = self.rel_pos.get_bias()
#             elif shared_rel_pos is not None:
#                 attn_bias = shared_rel_pos

#             x = torch.nn.functional.scaled_dot_product_attention(
#                 q.transpose(-1, -2).contiguous(),
#                 k.transpose(-1, -2).contiguous(),
#                 v.transpose(-1, -2).contiguous(),
#                 attn_mask=attn_bias,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             ).transpose(-1, -2).reshape(B, -1, H, W)
#         else:
#             q = q * self.scale
#             attn = q.transpose(-2, -1) @ k
#             if self.rel_pos is not None:
#                 attn = self.rel_pos(attn)
#             elif shared_rel_pos is not None:
#                 attn = attn + shared_rel_pos
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class AttentionCl(nn.Module):
#     """ Channels-last multi-head attention (B, ..., C) """
#     fused_attn: Final[bool]

#     def __init__(
#             self,
#             dim: int,
#             dim_out: Optional[int] = None,
#             dim_head: int = 32,
#             bias: bool = True,
#             expand_first: bool = True,
#             head_first: bool = True,
#             rel_pos_cls: Callable = None,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.
#     ):
#         super().__init__()
#         dim_out = dim_out or dim
#         dim_attn = dim_out if expand_first and dim_out > dim else dim
#         assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
#         self.num_heads = dim_attn // dim_head
#         self.dim_head = dim_head
#         self.head_first = head_first
#         self.scale = dim_head ** -0.5
#         self.fused_attn = use_fused_attn()

#         self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
#         self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, partition_block=True, shared_rel_pos: Optional[torch.Tensor] = None):
#         if partition_block:
#             x, rel_pos_row, rel_pos_col = x
#         B = x.shape[0]
#         restore_shape = x.shape[:-1]

#         if self.head_first:
#             q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
#         else:
#             q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

#         if self.fused_attn:
#             attn_bias = None
#             if self.rel_pos is not None:
#                 attn_bias = self.rel_pos.get_bias()
#             elif shared_rel_pos is not None:
#                 attn_bias = shared_rel_pos

#             x = torch.nn.functional.scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask=attn_bias,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             if self.rel_pos is not None:
#                 # num_heads, p^2, p^2
#                 bias = self.rel_pos.get_bias()
#                 if partition_block:
#                     row_bias = rel_pos_row.expand(self.num_heads, 1, -1)
#                     bias = torch.cat((row_bias, bias), dim=1)
#                     col_bias = rel_pos_col.expand(self.num_heads, -1, 1)
#                     bias = torch.cat((col_bias, bias), dim=2)
#                 attn = attn + bias
#             elif shared_rel_pos is not None:
#                 attn = attn + shared_rel_pos
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(restore_shape + (-1,))
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class LayerScale(nn.Module):
#     def __init__(self, dim, init_values=1e-5, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(dim))

#     def forward(self, x):
#         gamma = self.gamma
#         return x.mul_(gamma) if self.inplace else x * gamma


# class LayerScale2d(nn.Module):
#     def __init__(self, dim, init_values=1e-5, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(dim))

#     def forward(self, x):
#         gamma = self.gamma.view(1, -1, 1, 1)
#         return x.mul_(gamma) if self.inplace else x * gamma


# class Downsample2d(nn.Module):
#     """ A downsample pooling module supporting several maxpool and avgpool modes
#     * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
#     * 'max2' - MaxPool2d w/ kernel_size = stride = 2
#     * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
#     * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
#     """

#     def __init__(
#             self,
#             dim: int,
#             dim_out: int,
#             pool_type: str = 'avg2',
#             padding: str = '',
#             bias: bool = True,
#     ):
#         super().__init__()
#         assert pool_type in ('max', 'max2', 'avg', 'avg2')
#         if pool_type == 'max':
#             self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
#         elif pool_type == 'max2':
#             self.pool = create_pool2d('max', 2, padding=padding or 0)  # kernel_size == stride == 2
#         elif pool_type == 'avg':
#             self.pool = create_pool2d(
#                 'avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
#         else:
#             self.pool = create_pool2d('avg', 2, padding=padding or 0)

#         if dim != dim_out:
#             self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
#         else:
#             self.expand = nn.Identity()

#     def forward(self, x):
#         x = self.pool(x)  # spatial downsample
#         x = self.expand(x)  # expand chs
#         return x


# def _init_conv(module, name, scheme=''):
#     if isinstance(module, nn.Conv2d):
#         if scheme == 'normal':
#             nn.init.normal_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'trunc_normal':
#             trunc_normal_tf_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'xavier_normal':
#             nn.init.xavier_normal_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         else:
#             # efficientnet like
#             fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
#             fan_out //= module.groups
#             nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)


# def num_groups(group_size, channels):
#     if not group_size:  # 0 or None
#         return 1  # normal conv with 1 group
#     else:
#         # NOTE group_size == 1 -> depthwise conv
#         assert channels % group_size == 0
#         return channels // group_size


# class MbConvBlock(nn.Module):
#     """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
#     """
#     def __init__(
#             self,
#             in_chs: int,
#             out_chs: int,
#             stride: int = 1,
#             dilation: Tuple[int, int] = (1, 1),
#             cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
#             drop_path: float = 0.
#     ):
#         super(MbConvBlock, self).__init__()
#         norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
#         mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
#         groups = num_groups(cfg.group_size, mid_chs)

#         if stride == 2:
#             self.shortcut = Downsample2d(
#                 in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding)
#         else:
#             self.shortcut = nn.Identity()

#         assert cfg.stride_mode in ('pool', '1x1', 'dw')
#         stride_pool, stride_1, stride_2 = 1, 1, 1
#         if cfg.stride_mode == 'pool':
#             # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
#             stride_pool, dilation_2 = stride, dilation[1]
#             # FIXME handle dilation of avg pool
#         elif cfg.stride_mode == '1x1':
#             # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
#             stride_1, dilation_2 = stride, dilation[1]
#         else:
#             stride_2, dilation_2 = stride, dilation[0]

#         self.pre_norm = nn.BatchNorm2d(in_chs)
#         # self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
#         if stride_pool > 1:
#             self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding)
#         else:
#             self.down = nn.Identity()
#         self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
#         # self.norm1 = norm_act_layer(mid_chs)
#         self.norm1 = nn.BatchNorm2d(mid_chs)
#         self.act1 = nn.GELU()

#         self.conv2_kxk = create_conv2d(
#             mid_chs, mid_chs, cfg.kernel_size,
#             stride=stride_2, dilation=dilation_2, groups=groups, padding=cfg.padding)

#         attn_kwargs = {}
#         if isinstance(cfg.attn_layer, str):
#             if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
#                 attn_kwargs['act_layer'] = cfg.attn_act_layer
#                 attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

#         # two different orderings for SE and norm2 (due to some weights and trials using SE before norm2)
#         if cfg.attn_early:
#             self.se_early = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)
#             self.norm2 = norm_act_layer(mid_chs)
#             self.se = None
#         else:
#             self.se_early = None
#             # self.norm2 = norm_act_layer(mid_chs)
#             self.norm2 = nn.BatchNorm2d(mid_chs)
#             self.act2 = nn.GELU()
#             self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

#         self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_conv, scheme=scheme), self)

#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         x = self.pre_norm(x)
#         x = self.down(x)

#         # 1x1 expansion conv & norm-act
#         x = self.conv1_1x1(x)
#         x = self.act1(self.norm1(x))

#         # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
#         x = self.conv2_kxk(x)
#         if self.se_early is not None:
#             x = self.se_early(x)
#         x = self.act2(self.norm2(x))
#         if self.se is not None:
#             x = self.se(x)

#         # 1x1 linear projection to output width
#         x = self.conv3_1x1(x)
#         x = self.drop_path(x) + shortcut
#         return x


# def window_partition(x, window_size: List[int], agent=None):
#     B, H, W, C = x.shape
#     _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
#     _assert(W % window_size[1] == 0, '')
#     x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
#     agent = agent.view(B, H // window_size[0], 1, W // window_size[1], 1, C)
#     agent = agent.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, 1, C)
#     windows = torch.cat((agent, windows), dim=1)
#     return windows

# # def window_partition(x, window_size: List[int], agent=None):
# #     B, H, W, C = x.shape
# #     _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
# #     _assert(W % window_size[1] == 0, '')
# #     x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
# #     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
# #     if agent is not None:
# #         windows = windows.view(windows.shape[0], -1, C)
# #         # if agent.shape[0] == 1: agent = agent.view(1, 1, 1, C).expand(B, H//window_size[0], W//window_size[1], C)
# #         agent = agent.view(windows.shape[0], -1, C)
# #         windows = torch.cat((agent, windows), dim=1)
# #     return windows



# @register_notrace_function  # reason: int argument is a Proxy
# def window_reverse(windows, window_size: List[int], img_size: List[int], use_agent=False):
#     H, W = img_size
#     C = windows.shape[-1]

#     agent, windows = torch.split(windows, [1, windows.shape[-2]-1], dim=-2)

#     x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
#     agent = agent.view(-1, H // window_size[0], W // window_size[1], 1, 1, C)
#     agent = agent.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H // window_size[0], W // window_size[1], C)
#     return (x, agent)

# # @register_notrace_function  # reason: int argument is a Proxy
# # def window_reverse(windows, window_size: List[int], img_size: List[int], use_agent=False):
# #     H, W = img_size
# #     C = windows.shape[-1]
# #     if not use_agent:
# #         x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
# #         x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
# #         return x, None
# #     agent, windows = torch.split(windows, [1, windows.shape[-2]-1], dim=-2)
# #     x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
# #     agent = agent.view(-1, H // window_size[0], W // window_size[1], C)
# #     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
# #     return (x, agent)


# def get_rel_pos_cls(cfg: MaxxVitTransformerCfg, window_size):
#     rel_pos_cls = None
#     if cfg.rel_pos_type == 'mlp':
#         rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)
#     elif cfg.rel_pos_type == 'bias':
#         rel_pos_cls = partial(RelPosBias, window_size=window_size)
#     elif cfg.rel_pos_type == 'bias_tf':
#         rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
#     return rel_pos_cls

# def grid_partition(x, grid_size: List[int]):
#     B, H, W, C = x.shape
#     _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
#     _assert(W % grid_size[1] == 0, '')
#     x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
#     windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
#     return windows


# @register_notrace_function  # reason: int argument is a Proxy
# def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
#     H, W = img_size
#     C = windows.shape[-1]
#     x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
#     x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
#     return x

# class PartitionAttentionCl(nn.Module):
#     """ Grid or Block partition + Attn + FFN.
#     NxC 'channels last' tensor layout.
#     """

#     def __init__(
#             self,
#             dim: int,
#             partition_type: str = 'block',
#             cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path: float = 0.,
#             stage=0
#     ):
#         super().__init__()
#         norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
#         act_layer = get_act_layer(cfg.act_layer)

#         self.partition_block = partition_type == 'block'
        
#         self.agent_size = [8, 4, 2, 1]
#         # self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)

#         self.partition_size = to_2tuple(cfg.window_size if self.partition_block else self.agent_size[stage])
#         rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

#         self.norm1 = norm_layer(dim)
#         self.attn = AttentionCl(
#             dim,
#             dim,
#             dim_head=cfg.dim_head,
#             bias=cfg.attn_bias,
#             head_first=cfg.head_first,
#             rel_pos_cls=rel_pos_cls,
#             attn_drop=cfg.attn_drop,
#             proj_drop=cfg.proj_drop,
#         )
#         self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=int(dim * cfg.expand_ratio),
#             act_layer=act_layer,
#             drop=cfg.proj_drop)
#         self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.stage = stage

#     def _partition_attn(self, x, agent=None):
#         img_size = x.shape[1:3]
#         if agent is not None:
#             agent, rel_pos_row, rel_pos_col = agent
#         if self.partition_block:
#             partitioned = window_partition(x, self.partition_size, agent)
#         else:
#             if agent is not None:
#                 partitioned = x
#             else:
#                 partitioned = grid_partition(x, self.partition_size)

#         if agent is not None:
#             partitioned = self.attn((partitioned, rel_pos_row, rel_pos_col), self.partition_block)
#         else:
#             partitioned = self.attn(partitioned, self.partition_block)

#         if self.partition_block:
#             x, agent = window_reverse(partitioned, self.partition_size, img_size, use_agent=True)
#         else:
#             if agent is not None:
#                 x = partitioned
#             else:
#                 x = grid_reverse(partitioned, self.partition_size, img_size)
#         return (x, agent)

#     def forward(self, x):
#         if isinstance(x, tuple):
#             x, agent = x
#             agent, rel_pos_row, rel_pos_col = agent
#             y, yagent = self._partition_attn(self.norm1(x), (self.norm1(agent), rel_pos_row, rel_pos_col))
            
#             x = x + self.drop_path1(self.ls1(y))
#             x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

#             agent = agent + self.drop_path1(self.ls1(yagent))
#             agent = agent + self.drop_path2(self.ls2(self.mlp(self.norm2(agent))))

#             return x, agent
#         else:
#             # for agent
#             y, _ = self._partition_attn(self.norm1(x))
#             x = x + self.drop_path1(self.ls1(y))
#             x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x


# def window_partition_nchw(x, window_size: List[int]):
#     B, C, H, W = x.shape
#     _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
#     _assert(W % window_size[1] == 0, '')
#     x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
#     windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
#     return windows


# @register_notrace_function  # reason: int argument is a Proxy
# def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):
#     H, W = img_size
#     C = windows.shape[1]
#     x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
#     x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
#     return x


# def grid_partition_nchw(x, grid_size: List[int]):
#     B, C, H, W = x.shape
#     _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
#     _assert(W % grid_size[1] == 0, '')
#     x = x.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
#     windows = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, C, grid_size[0], grid_size[1])
#     return windows


# @register_notrace_function  # reason: int argument is a Proxy
# def grid_reverse_nchw(windows, grid_size: List[int], img_size: List[int]):
#     H, W = img_size
#     C = windows.shape[1]
#     x = windows.view(-1, H // grid_size[0], W // grid_size[1], C, grid_size[0], grid_size[1])
#     x = x.permute(0, 3, 4, 1, 5, 2).contiguous().view(-1, C, H, W)
#     return x

# class PartitionAttention2d(nn.Module):
#     """ Grid or Block partition + Attn + FFN

#     '2D' NCHW tensor layout.
#     """

#     def __init__(
#             self,
#             dim: int,
#             partition_type: str = 'block',
#             cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path: float = 0.,
#     ):
#         super().__init__()
#         norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)  # NOTE this block is channels-last
#         act_layer = get_act_layer(cfg.act_layer)

#         self.partition_block = partition_type == 'block'
#         self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
#         rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

#         self.norm1 = norm_layer(dim)
#         self.attn = Attention2d(
#             dim,
#             dim,
#             dim_head=cfg.dim_head,
#             bias=cfg.attn_bias,
#             head_first=cfg.head_first,
#             rel_pos_cls=rel_pos_cls,
#             attn_drop=cfg.attn_drop,
#             proj_drop=cfg.proj_drop,
#         )
#         self.ls1 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = ConvMlp(
#             in_features=dim,
#             hidden_features=int(dim * cfg.expand_ratio),
#             act_layer=act_layer,
#             drop=cfg.proj_drop)
#         self.ls2 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def _partition_attn(self, x):
#         img_size = x.shape[-2:]
#         if self.partition_block:
#             partitioned = window_partition_nchw(x, self.partition_size)
#         else:
#             partitioned = grid_partition_nchw(x, self.partition_size)

#         partitioned = self.attn(partitioned)

#         if self.partition_block:
#             x = window_reverse_nchw(partitioned, self.partition_size, img_size)
#         else:
#             x = grid_reverse_nchw(partitioned, self.partition_size, img_size)
#         return x

#     def forward(self, x):
#         x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x



# def _init_transformer(module, name, scheme=''):
#     if isinstance(module, (nn.Conv2d, nn.Linear)):
#         if scheme == 'normal':
#             nn.init.normal_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'trunc_normal':
#             trunc_normal_tf_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'xavier_normal':
#             nn.init.xavier_normal_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         else:
#             # vit like
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 if 'mlp' in name:
#                     nn.init.normal_(module.bias, std=1e-6)
#                 else:
#                     nn.init.zeros_(module.bias)

# class TransformerBlock2d(nn.Module):
#     """ Transformer block with 2D downsampling
#     '2D' NCHW tensor layout

#     Some gains can be seen on GPU using a 1D / CL block, BUT w/ the need to switch back/forth to NCHW
#     for spatial pooling, the benefit is minimal so ended up using just this variant for CoAt configs.

#     This impl was faster on TPU w/ PT XLA than the 1D experiment.
#     """

#     def __init__(
#             self,
#             dim: int,
#             dim_out: int,
#             stride: int = 1,
#             rel_pos_cls: Callable = None,
#             cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path: float = 0.,
#     ):
#         super().__init__()
#         norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
#         act_layer = get_act_layer(cfg.act_layer)

#         if stride == 2:
#             self.shortcut = Downsample2d(dim, dim_out, pool_type=cfg.pool_type, bias=cfg.shortcut_bias)
#             self.norm1 = nn.Sequential(OrderedDict([
#                 ('norm', norm_layer(dim)),
#                 ('down', Downsample2d(dim, dim, pool_type=cfg.pool_type)),
#             ]))
#         else:
#             assert dim == dim_out
#             self.shortcut = nn.Identity()
#             self.norm1 = norm_layer(dim)

#         self.attn = Attention2d(
#             dim,
#             dim_out,
#             dim_head=cfg.dim_head,
#             expand_first=cfg.expand_first,
#             bias=cfg.attn_bias,
#             rel_pos_cls=rel_pos_cls,
#             attn_drop=cfg.attn_drop,
#             proj_drop=cfg.proj_drop
#         )
#         self.ls1 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim_out)
#         self.mlp = ConvMlp(
#             in_features=dim_out,
#             hidden_features=int(dim_out * cfg.expand_ratio),
#             act_layer=act_layer,
#             drop=cfg.proj_drop)
#         self.ls2 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_transformer, scheme=scheme), self)

#     def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
#         x = self.shortcut(x) + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x

# class MaxxVitBlock(nn.Module):
#     """ MaxVit conv, window partition + FFN , grid partition + FFN
#     """

#     def __init__(
#             self,
#             dim: int,
#             dim_out: int,
#             stride: int = 1,
#             conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
#             transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path: float = 0.,
#             stage=0
#     ):
#         super().__init__()
#         # False
#         self.nchw_attn = transformer_cfg.use_nchw_attn

#         conv_cls = MbConvBlock
#         self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
#         attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
#         partition_layer = PartitionAttention2d if self.nchw_attn else PartitionAttentionCl
#         self.attn_block = None if transformer_cfg.no_block_attn else partition_layer(stage=stage, **attn_kwargs)
#         self.attn_grid = partition_layer(partition_type='agent', stage=stage, **attn_kwargs)
        
#         self.stage = stage

#     def init_weights(self, scheme=''):
#         if self.attn_block is not None:
#             named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
#         named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
#         named_apply(partial(_init_conv, scheme=scheme), self.conv)

#     def forward(self, x):
#         # NCHW format
#         x, agent = x
#         # if agent is not None:
#         #     stage_size = [8, 4, 2, 1]
#         #     agent = agent.expand(x.shape[0], -1, stage_size[self.stage], stage_size[self.stage])
#         agent, rel_pos_row, rel_pos_col = agent
#         if self.stage == 0:
#             agent = agent.expand(x.shape[0], -1, 8, 8)
#         # else:
#         #     agent = self.conv(agent)    
#         x = self.conv(x)
        

#         if not self.nchw_attn:
#             x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
#             agent = agent.permute(0, 2, 3, 1)
#         if self.attn_block is not None:
#             x, agent = self.attn_block((x, (agent, rel_pos_row, rel_pos_col)))
#         agent = self.attn_grid(agent)
#         if not self.nchw_attn:
#             x = x.permute(0, 3, 1, 2)  # back to NCHW
#             agent = agent.permute(0, 3, 1, 2)
#         return (x, (agent, rel_pos_row, rel_pos_col))


# class MaxxVitStage(nn.Module):
#     def __init__(
#             self,
#             in_chs: int,
#             out_chs: int,
#             stride: int = 2,
#             depth: int = 4,
#             feat_size: Tuple[int, int] = (14, 14),
#             block_types: Union[str, Tuple[str]] = 'C',
#             transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
#             drop_path: Union[float, List[float]] = 0.,
#             stage=0
#     ):
#         super().__init__()
#         self.grad_checkpointing = False

#         self.projection = nn.Linear(in_chs, out_chs)

#         block_types = extend_tuple(block_types, depth)
#         blocks = []
#         for i, t in enumerate(block_types):
#             block_stride = stride if i == 0 else 1
#             assert t in ('C', 'T', 'M')
#             if t == 'C':
#                 conv_cls = MbConvBlock
#                 blocks += [conv_cls(
#                     in_chs,
#                     out_chs,
#                     stride=block_stride,
#                     cfg=conv_cfg,
#                     drop_path=drop_path[i],
#                 )]
#             elif t == 'T':
#                 rel_pos_cls = get_rel_pos_cls(transformer_cfg, feat_size)
#                 blocks += [TransformerBlock2d(
#                     in_chs,
#                     out_chs,
#                     stride=block_stride,
#                     rel_pos_cls=rel_pos_cls,
#                     cfg=transformer_cfg,
#                     drop_path=drop_path[i],
#                 )]
#             elif t == 'M':
#                 blocks += [MaxxVitBlock(
#                     in_chs,
#                     out_chs,
#                     stride=block_stride,
#                     conv_cfg=conv_cfg,
#                     transformer_cfg=transformer_cfg,
#                     drop_path=drop_path[i],
#                     stage=stage
#                 )]
#             in_chs = out_chs
#         self.blocks = nn.Sequential(*blocks)

#         self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.stage = stage

#     def forward(self, x):
#         if self.grad_checkpointing and not torch.jit.is_scripting():
#             x = checkpoint_seq(self.blocks, x)
#         else:
#             # tuple (x, agent)
#             x, agent, agentList = x
#             agent, row, col = agent
#             agent = self.projection(agent.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#             agent = (agent, row, col)
#             x, agent = self.blocks((x, agent))
#             agent, row, col = agent
#             if agent.shape[-1] > 1:
#                 agent = self.maxPooling(agent)
#         if self.stage < 3: 
#             return (x, (agent, row, col), agentList)
#         return x


# class Stem(nn.Module):

#     def __init__(
#             self,
#             in_chs: int,
#             out_chs: int,
#             kernel_size: int = 3,
#             padding: str = '',
#             bias: bool = False,
#             act_layer: str = 'gelu',
#             norm_layer: str = 'batchnorm2d',
#             norm_eps: float = 1e-5,
#     ):
#         super().__init__()
#         if not isinstance(out_chs, (list, tuple)):
#             out_chs = to_2tuple(out_chs)

#         norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
#         self.out_chs = out_chs[-1]
#         self.stride = 2

#         self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
#         # self.norm1 = norm_act_layer(out_chs[0])
#         self.norm1 = nn.BatchNorm2d(out_chs[0], eps=norm_eps)
#         self.act1 = nn.GELU()
#         self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_conv, scheme=scheme), self)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act1(self.norm1(x))
#         x = self.conv2(x)
#         return x




# def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
#     if cfg.window_size is not None:
#         assert cfg.grid_size
#         return cfg
#     # 7 x 7
#     partition_size = img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio
#     cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
#     return cfg


# def _overlay_kwargs(cfg: MaxxVitCfg, **kwargs):
#     transformer_kwargs = {}
#     conv_kwargs = {}
#     base_kwargs = {}
#     for k, v in kwargs.items():
#         if k.startswith('transformer_'):
#             transformer_kwargs[k.replace('transformer_', '')] = v
#         elif k.startswith('conv_'):
#             conv_kwargs[k.replace('conv_', '')] = v
#         else:
#             base_kwargs[k] = v
#     cfg = replace(
#         cfg,
#         transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
#         conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
#         **base_kwargs
#     )
#     return cfg


# class MaxxVit(nn.Module):
#     """ CoaTNet + MaxVit base model.

#     Highly configurable for different block compositions, tensor layouts, pooling types.
#     """

#     def __init__(
#             self,
#             cfg: MaxxVitCfg,
#             img_size: Union[int, Tuple[int, int]] = 224,
#             in_chans: int = 3,
#             num_classes: int = 1000,
#             global_pool: str = 'avg',
#             drop_rate: float = 0.,
#             drop_path_rate: float = 0.,
#             **kwargs,
#     ):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         if kwargs:
#             cfg = _overlay_kwargs(cfg, **kwargs)
#         transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
#         self.num_classes = num_classes
#         self.global_pool = global_pool
#         self.num_features = self.embed_dim = cfg.embed_dim[-1]
#         self.drop_rate = drop_rate
#         self.grad_checkpointing = False
#         self.feature_info = []

#         self.stem = Stem(
#             in_chs=in_chans,
#             out_chs=cfg.stem_width,
#             padding=cfg.conv_cfg.padding,
#             bias=cfg.stem_bias,
#             act_layer=cfg.conv_cfg.act_layer,
#             norm_layer=cfg.conv_cfg.norm_layer,
#             norm_eps=cfg.conv_cfg.norm_eps,
#         )
#         stride = self.stem.stride
#         self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
#         feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

#         num_stages = len(cfg.embed_dim)
#         assert len(cfg.depths) == num_stages
#         dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
#         in_chs = self.stem.out_chs
#         stages = []
#         for i in range(num_stages):
#             stage_stride = 2
#             out_chs = cfg.embed_dim[i]
#             feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
#             stages += [MaxxVitStage(
#                 in_chs,
#                 out_chs,
#                 depth=cfg.depths[i],
#                 block_types=cfg.block_type[i],
#                 conv_cfg=cfg.conv_cfg,
#                 transformer_cfg=transformer_cfg,
#                 feat_size=feat_size,
#                 drop_path=dpr[i],
#                 stage=i
#             )]
#             stride *= stage_stride
#             in_chs = out_chs
#             self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
#         self.stages = nn.Sequential(*stages)

#         final_norm_layer = partial(get_norm_layer(cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
#         self.head_hidden_size = cfg.head_hidden_size
#         if self.head_hidden_size:
#             self.norm = nn.Identity()
#             self.head = NormMlpClassifierHead(
#                 self.num_features,
#                 num_classes,
#                 hidden_size=self.head_hidden_size,
#                 pool_type=global_pool,
#                 drop_rate=drop_rate,
#                 norm_layer=final_norm_layer,
#             )
#         else:
#             # standard classifier head w/ norm, pooling, fc classifier
#             self.norm = final_norm_layer(self.num_features)
#             self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

#         self.agent_norm0 = final_norm_layer(cfg.embed_dim[0])
#         self.agent_head0 = ClassifierHead(cfg.embed_dim[0], num_classes, pool_type=global_pool, drop_rate=drop_rate)
#         self.agent_norm1 = final_norm_layer(cfg.embed_dim[1])
#         self.agent_head1 = ClassifierHead(cfg.embed_dim[1], num_classes, pool_type=global_pool, drop_rate=drop_rate)
#         self.agent_norm2 = final_norm_layer(cfg.embed_dim[2])
#         self.agent_head2 = ClassifierHead(cfg.embed_dim[2], num_classes, pool_type=global_pool, drop_rate=drop_rate)
#         self.agent_norm3 = final_norm_layer(cfg.embed_dim[3])
#         self.agent_head3 = ClassifierHead(cfg.embed_dim[3], num_classes, pool_type=global_pool, drop_rate=drop_rate)


#         self.agent = nn.Parameter(torch.zeros(1, 64, 1, 1))
#         nn.init.normal_(self.agent, std=1e-6)
#         self.global_pos_row = nn.Parameter(torch.zeros(1, 1, 49))
#         self.global_pos_col = nn.Parameter(torch.zeros(1, 50, 1))
#         self.prob, self.label = [], []
        
#         self.video_name = []
#         # Weight init (default PyTorch init works well for AdamW if scheme not set)
#         assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
#         if cfg.weight_init:
#             named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

#     def _init_weights(self, module, name, scheme=''):
#         if hasattr(module, 'init_weights'):
#             try:
#                 module.init_weights(scheme=scheme)
#             except TypeError:
#                 module.init_weights()

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {
#             k for k, _ in self.named_parameters()
#             if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

#     @torch.jit.ignore
#     def group_matcher(self, coarse=False):
#         matcher = dict(
#             stem=r'^stem',  # stem and embed
#             blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
#         )
#         return matcher

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         for s in self.stages:
#             s.grad_checkpointing = enable

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head.fc

#     def reset_classifier(self, num_classes, global_pool=None):
#         self.num_classes = num_classes
#         self.head.reset(num_classes, global_pool)

#     def forward_features(self, x):
#         x = self.stem(x)
        
#         x = self.stages((x, (self.agent, self.global_pos_row, self.global_pos_col), []))
#         x = self.norm(x)

#         return x

#     def forward_head(self, x, pre_logits: bool = False):
#         return self.head(x, pre_logits=pre_logits)

#     def forward(self, x, inference=False):
        
#         x = self.forward_features(x)
#         sigmoid = nn.Sigmoid()
#         logits = sigmoid(self.forward_head(x))

#         return logits


# def _create_maxxvit(variant, cfg_variant=None, pretrained=False, **kwargs):
#     if cfg_variant is None:
#         if variant in model_cfgs:
#             cfg_variant = variant
#         else:
#             cfg_variant = '_'.join(variant.split('_')[:-1])
#     return build_model_with_cfg(
#         MaxxVit, variant, pretrained,
#         model_cfg=model_cfgs[cfg_variant],
#         feature_cfg=dict(flatten_sequential=True),
#         pretrained_filter_fn=checkpoint_filter_fn,
#         **kwargs)


# @register_model
# def vis_agentViT_base_tf_224(pretrained=False, **kwargs) -> MaxxVit:
#     return _create_maxxvit('maxvit_base_tf_224', 'maxvit_base_tf', pretrained=pretrained, **kwargs)





























# def _tf_cfg():
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             norm_eps=1e-3,
#             act_layer='gelu_tanh',
#             padding='same',
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             norm_eps=1e-5,
#             act_layer='gelu_tanh',
#             head_first=False,  # heads are interleaved (q_nh, q_hdim, k_nh, q_hdim, ....)
#             rel_pos_type='bias_tf',
#         ),
#     )

# def _rw_coat_cfg(
#         stride_mode='pool',
#         pool_type='avg2',
#         conv_output_bias=False,
#         conv_attn_early=False,
#         conv_attn_act_layer='relu',
#         conv_norm_layer='',
#         transformer_shortcut_bias=True,
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         init_values=None,
#         rel_pos_type='bias',
#         rel_pos_dim=512,
# ):
#     # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
#     # Common differences for initial timm models:
#     # - pre-norm layer in MZBConv included an activation after norm
#     # - mbconv expansion calculated from input instead of output chs
#     # - mbconv shortcut and final 1x1 conv did not have a bias
#     # - SE act layer was relu, not silu
#     # - mbconv uses silu in timm, not gelu
#     # - expansion in attention block done via output proj, not input proj
#     # Variable differences (evolved over training initial models):
#     # - avg pool with kernel_size=2 favoured downsampling (instead of maxpool for coat)
#     # - SE attention was between conv2 and norm/act
#     # - default to avg pool for mbconv downsample instead of 1x1 or dw conv
#     # - transformer block shortcut has no bias
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             pre_norm_act=True,
#             expand_output=False,
#             output_bias=conv_output_bias,
#             attn_early=conv_attn_early,
#             attn_act_layer=conv_attn_act_layer,
#             act_layer='silu',
#             norm_layer=conv_norm_layer,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             shortcut_bias=transformer_shortcut_bias,
#             pool_type=pool_type,
#             init_values=init_values,
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )


# def _rw_max_cfg(
#         stride_mode='dw',
#         pool_type='avg2',
#         conv_output_bias=False,
#         conv_attn_ratio=1 / 16,
#         conv_norm_layer='',
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         window_size=None,
#         dim_head=32,
#         init_values=None,
#         rel_pos_type='bias',
#         rel_pos_dim=512,
# ):
#     # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
#     # Differences of initial timm models:
#     # - mbconv expansion calculated from input instead of output chs
#     # - mbconv shortcut and final 1x1 conv did not have a bias
#     # - mbconv uses silu in timm, not gelu
#     # - expansion in attention block done via output proj, not input proj
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             expand_output=False,
#             output_bias=conv_output_bias,
#             attn_ratio=conv_attn_ratio,
#             act_layer='silu',
#             norm_layer=conv_norm_layer,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             pool_type=pool_type,
#             dim_head=dim_head,
#             window_size=window_size,
#             init_values=init_values,
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )


# def _next_cfg(
#         stride_mode='dw',
#         pool_type='avg2',
#         conv_norm_layer='layernorm2d',
#         conv_norm_layer_cl='layernorm',
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         window_size=None,
#         no_block_attn=False,
#         init_values=1e-6,
#         rel_pos_type='mlp',  # MLP by default for maxxvit
#         rel_pos_dim=512,
# ):
#     # For experimental models with convnext instead of mbconv
#     init_values = to_2tuple(init_values)
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             block_type='convnext',
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             expand_output=False,
#             init_values=init_values[0],
#             norm_layer=conv_norm_layer,
#             norm_layer_cl=conv_norm_layer_cl,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             pool_type=pool_type,
#             window_size=window_size,
#             no_block_attn=no_block_attn,  # enabled for MaxxViT-V2
#             init_values=init_values[1],
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )

# model_cfgs = dict(

#     maxvit_tiny_tf=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=512,
#         **_tf_cfg(),
#     ),
#     maxvit_small_tf=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=768,
#         **_tf_cfg(),
#     ),
#     maxvit_base_tf=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 8, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=768,
#         **_tf_cfg(),
#     ),
# )


# def checkpoint_filter_fn(state_dict, model: nn.Module):
#     model_state_dict = model.state_dict()
#     out_dict = {}
#     for k, v in state_dict.items():
#         if k.endswith('relative_position_bias_table'):
#             m = model.get_submodule(k[:-29])
#             if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
#                 v = resize_rel_pos_bias_table(
#                     v,
#                     new_window_size=m.window_size,
#                     new_bias_shape=m.relative_position_bias_table.shape,
#                 )

#         if k in model_state_dict and v.ndim != model_state_dict[k].ndim and v.numel() == model_state_dict[k].numel():
#             # adapt between conv2d / linear layers
#             assert v.ndim in (2, 4)
#             v = v.reshape(model_state_dict[k].shape)
#         out_dict[k] = v
#     return out_dict


import math
from collections import OrderedDict
from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, Optional, Union, Tuple, List

import torch
from torch import nn
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, ConvMlp, DropPath, LayerNorm, ClassifierHead, NormMlpClassifierHead
from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
from timm.layers import trunc_normal_tf_, to_2tuple, extend_tuple, make_divisible, _assert, SelectAdaptivePool2d
from timm.layers import RelPosMlp, RelPosBias, RelPosBiasTf, use_fused_attn, resize_rel_pos_bias_table
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model

import torch.distributed as dist


@dataclass
class MaxxVitTransformerCfg:
    dim_head: int = 32
    head_first: bool = True  # head ordering in qkv channel dim
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = None
    grid_size: Optional[Tuple[int, int]] = None
    no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
    use_nchw_attn: bool = False  # for MaxViT variants (not used for CoAt), keep tensors in NCHW order
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class MaxxVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = ''
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class MaxxVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = False
    conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)
    transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)
    head_hidden_size: int = None
    weight_init: str = 'vit_eff'


class Attention2d(nn.Module):
    fused_attn: Final[bool]

    """ multi-head attention for 2D NCHW tensors"""
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape

        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

        if self.fused_attn:
            attn_bias = None
            if self.rel_pos is not None:
                attn_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                attn_bias = shared_rel_pos

            x = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2).contiguous(),
                k.transpose(-1, -2).contiguous(),
                v.transpose(-1, -2).contiguous(),
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-1, -2).reshape(B, -1, H, W)
        else:
            q = q * self.scale
            attn = q.transpose(-2, -1) @ k
            if self.rel_pos is not None:
                attn = self.rel_pos(attn)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.c_attn = nn.Parameter(torch.ones((self.num_heads,)), requires_grad=True)
        


    def forward(self, x, partition_block=True, shared_rel_pos: Optional[torch.Tensor] = None):
        if partition_block:
            x, rel_pos_row, rel_pos_col = x
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel_pos is not None:
            # num_heads, p^2, p^2
            bias = self.rel_pos.get_bias()
            if partition_block:
                row_bias = rel_pos_row.expand(self.num_heads, -1, 49)
                bias = torch.cat((row_bias, bias), dim=1)
                col_bias = rel_pos_col.expand(self.num_heads, 49+row_bias.shape[1], -1)
                bias = torch.cat((col_bias, bias), dim=2)
            attn = attn + bias
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class Downsample2d(nn.Module):
    """ A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            padding: str = '',
            bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        if pool_type == 'max':
            self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
        elif pool_type == 'max2':
            self.pool = create_pool2d('max', 2, padding=padding or 0)  # kernel_size == stride == 2
        elif pool_type == 'avg':
            self.pool = create_pool2d(
                'avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
        else:
            self.pool = create_pool2d('avg', 2, padding=padding or 0)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'zeros':
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MbConvBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.
    ):
        super(MbConvBlock, self).__init__()
        norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)

        if stride == 2:
            self.shortcut = Downsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding)
        elif in_chs == out_chs:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_chs, out_chs, kernel_size=1)

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        self.pre_norm = nn.BatchNorm2d(in_chs)
        # self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        if stride_pool > 1:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
        # self.norm1 = norm_act_layer(mid_chs)
        self.norm1 = nn.BatchNorm2d(mid_chs)
        self.act1 = nn.GELU()

        self.conv2_kxk = create_conv2d(
            mid_chs, mid_chs, cfg.kernel_size,
            stride=stride_2, dilation=dilation_2, groups=groups, padding=cfg.padding)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        # two different orderings for SE and norm2 (due to some weights and trials using SE before norm2)
        if cfg.attn_early:
            self.se_early = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)
            self.norm2 = norm_act_layer(mid_chs)
            self.se = None
        else:
            self.se_early = None
            # self.norm2 = norm_act_layer(mid_chs)
            self.norm2 = nn.BatchNorm2d(mid_chs)
            self.act2 = nn.GELU()
            self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = self.act1(self.norm1(x))

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.act2(self.norm2(x))
        if self.se is not None:
            x = self.se(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x


def window_partition(x, window_size: List[int], agent=None, agent_num=1):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    agent = agent.view(B, agent_num, H // window_size[0], 1, W // window_size[1], 1, C)
    agent = agent.permute(0, 2, 4, 3, 5, 1, 6).contiguous().view(-1, agent_num, C)
    windows = torch.cat((agent, windows), dim=1)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: List[int], img_size: List[int], agent_num=1):
    H, W = img_size
    C = windows.shape[-1]

    agent, windows = torch.split(windows, [agent_num, windows.shape[-2]-agent_num], dim=-2)

    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    agent = agent.contiguous().view(-1, agent_num, H // window_size[0], W // window_size[1], 1, 1, C)
    agent = agent.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, agent_num, H // window_size[0], W // window_size[1], C)
    return (x, agent)


def get_rel_pos_cls(cfg: MaxxVitTransformerCfg, window_size):
    rel_pos_cls = None
    if cfg.rel_pos_type == 'mlp':
        rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)
    elif cfg.rel_pos_type == 'bias':
        rel_pos_cls = partial(RelPosBias, window_size=window_size)
    elif cfg.rel_pos_type == 'bias_tf':
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls

def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x

class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
            stage=0
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)

        self.partition_block = partition_type == 'block'
        
        self.agent_size = [8, 4, 2, 1]
        # self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)

        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else self.agent_size[stage])
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer(dim)
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.stage = stage

        # self.addition_ln1 = norm_layer(dim)
        # self.addition_ln2 = norm_layer(dim)
        # self.addition_lna1 = norm_layer(dim)
        # self.addition_lna2 = norm_layer(dim)

    def _partition_attn(self, x, agent=None):
        img_size = x.shape[1:3]
        if agent is not None:
            agent, rel_pos_row, rel_pos_col = agent
        if self.partition_block:
            agent_num = agent.shape[1]
            partitioned = window_partition(x, self.partition_size, agent, agent_num)
        else:
            partitioned = x
            # if agent is not None:
            #     partitioned = x
            # else:
            #     partitioned = grid_partition(x, self.partition_size)

        if agent is not None:
            partitioned = self.attn((partitioned, rel_pos_row, rel_pos_col), self.partition_block).contiguous()
        else:
            partitioned = self.attn(partitioned, self.partition_block).contiguous()

        if self.partition_block:
            x, agent = window_reverse(partitioned, self.partition_size, img_size, agent_num=agent_num)
            x = x.contiguous()
            agent = agent.contiguous()
        else:
            x = partitioned.contiguous()
            # if agent is not None:
            #     x = partitioned
            # else:
            #     x = grid_reverse(partitioned, self.partition_size, img_size)
        return (x, agent)

    def forward(self, x):
        if isinstance(x, tuple):
            x, agent = x
            agent, rel_pos_row, rel_pos_col = agent
            y, yagent = self._partition_attn(self.norm1(x), (self.norm1(agent), rel_pos_row, rel_pos_col))
            
            x = x + self.drop_path1(self.ls1(y))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

            agent = agent + self.drop_path1(self.ls1(yagent))
            agent = agent + self.drop_path2(self.ls2(self.mlp(self.norm2(agent))))

            return x, agent
        else:
            # for agent
            y, _ = self._partition_attn(self.norm1(x))
            x = x + self.drop_path1(self.ls1(y))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def window_partition_nchw(x, window_size: List[int]):
    B, C, H, W = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def grid_partition_nchw(x, grid_size: List[int]):
    B, C, H, W = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    windows = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, C, grid_size[0], grid_size[1])
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse_nchw(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], C, grid_size[0], grid_size[1])
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous().view(-1, C, H, W)
    return x

class PartitionAttention2d(nn.Module):
    """ Grid or Block partition + Attn + FFN

    '2D' NCHW tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer(dim)
        self.attn = Attention2d(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[-2:]
        if self.partition_block:
            partitioned = window_partition_nchw(x, self.partition_size)
        else:
            partitioned = grid_partition_nchw(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse_nchw(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse_nchw(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



def _init_transformer(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # vit like
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)

class TransformerBlock2d(nn.Module):
    """ Transformer block with 2D downsampling
    '2D' NCHW tensor layout

    Some gains can be seen on GPU using a 1D / CL block, BUT w/ the need to switch back/forth to NCHW
    for spatial pooling, the benefit is minimal so ended up using just this variant for CoAt configs.

    This impl was faster on TPU w/ PT XLA than the 1D experiment.
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            rel_pos_cls: Callable = None,
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
        act_layer = get_act_layer(cfg.act_layer)

        if stride == 2:
            self.shortcut = Downsample2d(dim, dim_out, pool_type=cfg.pool_type, bias=cfg.shortcut_bias)
            self.norm1 = nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('down', Downsample2d(dim, dim, pool_type=cfg.pool_type)),
            ]))
        else:
            assert dim == dim_out
            self.shortcut = nn.Identity()
            self.norm1 = norm_layer(dim)

        self.attn = Attention2d(
            dim,
            dim_out,
            dim_head=cfg.dim_head,
            expand_first=cfg.expand_first,
            bias=cfg.attn_bias,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop
        )
        self.ls1 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = ConvMlp(
            in_features=dim_out,
            hidden_features=int(dim_out * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_transformer, scheme=scheme), self)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = self.shortcut(x) + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MaxxVitBlock(nn.Module):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
            stage=0
    ):
        super().__init__()
        # False
        self.nchw_attn = transformer_cfg.use_nchw_attn

        conv_cls = MbConvBlock
        self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
        if stage < 2:
            self.addition_conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
            self.addition_gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
            self.addition_act = nn.GELU()
            self.addition_fc = nn.Linear(dim_out, dim_out)
        else:
            self.addition_conv = None
        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        partition_layer = PartitionAttention2d if self.nchw_attn else PartitionAttentionCl
        self.attn_block = None if transformer_cfg.no_block_attn else partition_layer(stage=stage, **attn_kwargs)
        self.attn_grid = partition_layer(partition_type='agent', stage=stage, **attn_kwargs)
        
        self.stage = stage

    def init_weights(self, scheme=''):
        if self.attn_block is not None:
            named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
        named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
        named_apply(partial(_init_conv, scheme=scheme), self.conv)
        if self.addition_conv is not None:
            named_apply(partial(_init_conv, scheme='zeros'), self.addition_conv)


    def forward(self, x):
        # NCHW format
        x, agent = x
        
        agent, rel_pos_row, rel_pos_col = agent
        agent_num = agent.shape[1]
        if self.stage == 0:
            agent = agent.expand(x.shape[0], agent_num, -1, 8, 8)
            
        # if self.stage < 2:
        #     x_raw = self.conv(x)
        #     x = x_raw - self.addition_act(self.addition_fc(self.addition_conv(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        # else:
        x = self.conv(x)
        if not self.nchw_attn:
            x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
            agent = agent.permute(0, 1, 3, 4, 2)
        if self.attn_block is not None:
            x, agent = self.attn_block((x, (agent, rel_pos_row, rel_pos_col)))
        
        B, A, H, W, C = agent.shape
        agent = agent.view(-1, H, W, C)
        agent = self.attn_grid(agent)
        agent = agent.view(B, A, H, W, C)
        if not self.nchw_attn:
            x = x.permute(0, 3, 1, 2)  # back to NCHW
            agent = agent.permute(0, 1, 4, 2, 3)
        return (x, (agent, rel_pos_row, rel_pos_col))


class MaxxVitStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
            stage=0
    ):
        super().__init__()
        self.grad_checkpointing = False

        self.projection = nn.Linear(in_chs, out_chs)

        block_types = extend_tuple(block_types, depth)
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
            assert t in ('C', 'T', 'M')
            if t == 'C':
                conv_cls = MbConvBlock
                blocks += [conv_cls(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    cfg=conv_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'T':
                rel_pos_cls = get_rel_pos_cls(transformer_cfg, feat_size)
                blocks += [TransformerBlock2d(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    rel_pos_cls=rel_pos_cls,
                    cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'M':
                blocks += [MaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                    stage=stage
                )]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage = stage

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            # tuple (x, agent)
            x, agent, agentList = x
            agent, row, col = agent
            agent = self.projection(agent.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            agent = (agent, row, col)
            x, agent = self.blocks((x, agent))
            agent, row, col = agent
            agentList.append(agent)
            if agent.shape[-1] > 1:
                B, A, C, H, W = agent.shape
                agent = agent.view(-1, C, H, W)
                agent = self.maxPooling(agent)
                agent = agent.view(B, A, C, agent.shape[-2], agent.shape[-1])
        return (x, (agent, row, col), agentList)


class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2

        self.addition_conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.addition_conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)
        self.addition_gamma1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.addition_gamma2 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        # self.norm1 = norm_act_layer(out_chs[0])
        self.norm1 = nn.BatchNorm2d(out_chs[0], eps=norm_eps)
        self.act1 = nn.GELU()
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme='zeros'), self)

    def forward(self, x):
        # x1 = 
        # x2 = self.act1(self.norm1(self.addition_conv2(x1)))
        # x = self.conv1(x) + self.addition_gamma1 * self.addition_conv1(x)     # 015
        x = self.conv1(x)   # window
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        # x = self.conv2(x) + self.addition_gamma2 * self.addition_conv2(x)
        return x




def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
    # 7 x 7
    partition_size = img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio
    cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
    return cfg


def _overlay_kwargs(cfg: MaxxVitCfg, **kwargs):
    transformer_kwargs = {}
    conv_kwargs = {}
    base_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('transformer_'):
            transformer_kwargs[k.replace('transformer_', '')] = v
        elif k.startswith('conv_'):
            conv_kwargs[k.replace('conv_', '')] = v
        else:
            base_kwargs[k] = v
    cfg = replace(
        cfg,
        transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
        conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
        **base_kwargs
    )
    return cfg


class FeatureDisentangle(nn.Module):
    def __init__(
            self,
            dim=768,
            agent_num=3,
    ):
        super().__init__()
        self.spe_fc1 = nn.Linear(dim*agent_num, dim)
        self.spe_fc2 = nn.Linear(dim, dim)
        self.spe_fc3 = nn.Linear(dim, dim)
        self.spe_reset_gate = nn.Linear(dim, dim)

        self.com_fc1 = nn.Linear(dim*agent_num, dim)
        self.com_fc2 = nn.Linear(dim, dim)
        self.com_fc3 = nn.Linear(dim, dim)
        self.com_reset_gate = nn.Linear(dim, dim)

        self.act_layer = nn.GELU()
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x_spe0 = self.act_layer(self.spe_fc1(x))
        x_com0 = self.act_layer(self.com_fc1(x))
        temp_com = self.act_layer(self.spe_reset_gate(x_spe0)) * x_com0
        temp_spe = self.act_layer(self.com_reset_gate(x_com0)) * x_spe0

        x_com = temp_com
        x_spe = temp_spe

        x_com = self.act_layer(self.com_fc2(x_com))
        x_spe = self.act_layer(self.spe_fc2(x_spe))

        x_com = x_com0 + self.com_fc3(x_com)
        x_spe = x_spe0 + self.spe_fc3(x_spe)

        x_com = x_com.permute(0, 3, 1, 2)
        x_spe = x_spe.permute(0, 3, 1, 2)
        return x_com, x_spe

import torch.autograd as autograd

class GradientReversalLayer(autograd.Function):
    """
    自定义梯度反转层
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        正向传播，不做任何操作
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播，对梯度进行反转
        """
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    """
    梯度反转模块
    """
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class agentClassifierHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=384, num_classes=1, 
                 act_layer: Union[str, Callable] = 'tanh') -> None:
        super().__init__()

        linear_layer = nn.Linear
        act_layer = get_act_layer(act_layer)
        self.norm = nn.LayerNorm(dim)
        # self.base_linear = nn.Linear(dim, num_classes)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', linear_layer(dim, hidden_dim)),
            # ('norm', nn.LayerNorm(hidden_dim)),
            ('act', act_layer()),
            # ('fc1', linear_layer(hidden_dim, 512)),
            # ('norm1', nn.LayerNorm(512)),
            # ('act1', act_layer()),
            # ('fc2', linear_layer(512, 384)),
            # ('act2', act_layer()),
        ]))
        self.num_features = 384
        self.fc = nn.Linear(self.num_features, num_classes)
        self.act_layer = act_layer()
        # self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        
        if len(x.shape) == 4: x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.flatten(x)
        # raw_x = self.base_linear(x)
        x = self.pre_logits(x)
        # x = self.drop(x)
        x = self.fc(x)
        return x
    
class patchHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            act_layer: Union[str, Callable] = 'tanh',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = False
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', linear_layer(in_features, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if self.hidden_size:
            if ((isinstance(self.pre_logits.fc, nn.Conv2d) and not self.use_conv) or
                    (isinstance(self.pre_logits.fc, nn.Linear) and self.use_conv)):
                with torch.no_grad():
                    new_fc = linear_layer(self.in_features, self.hidden_size)
                    new_fc.weight.copy_(self.pre_logits.fc.weight.reshape(new_fc.weight.shape))
                    new_fc.bias.copy_(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        N, C, H, W = x.shape
        patch_x = x.permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
        patch_x = self.norm(patch_x)
        patch_x = self.flatten(patch_x)
        patch_x = self.pre_logits(patch_x)
        patch_x = self.drop(patch_x)
        patch_x = self.fc(patch_x)

        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x, self.fc(x), patch_x
        x = self.fc(x)
        return x, patch_x

# 第一版
# class SimilarityClassifier(nn.Module):
#     def __init__(self,
#                  conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()) -> None:
#         super().__init__()
        
        
#         self.norm = nn.LayerNorm(49*49)
        
#         self.fc = nn.Linear(128, 8)
#         self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(49*49, 128)),
#                 ('act', nn.Tanh()),
#                 # ('dropout', nn.Dropout(0.5))  # 添加一个 Dropout 层，丢弃概率为 0.5 第一版好像没用 dropout, 但是昨晚的测试有dropout
#             ]))
        
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.norm(x)
#         x = self.pre_logits(x)
#         x = self.fc(x)
#         return x

class SimilarityClassifier(nn.Module):
    def __init__(self,
                 dim=49,
                 ) -> None:
        super().__init__()
        
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # (B, C, C)
        self.fc1 = nn.Linear(dim, 1)
        self.fc2 = nn.Linear(dim, 2)
        self.act = nn.Tanh()
        # self.pre_logits = nn.Sequential(OrderedDict([
        #         ('fc', nn.Linear(64*64, 64)),
        #         ('act', nn.Tanh()),
        #         # ('dropout', nn.Dropout(0.5))  # 添加一个 Dropout 层，丢弃概率为 0.5 第一版好像没用 dropout, 但是昨晚的测试有dropout
        #     ]))
        
    def forward(self, x):
        B, C, _ = x.size()
        x = x.view(B*C, C)
        x = self.norm1(x)
        x = self.fc1(x)
        x = x.view(B, C)
        x = self.norm2(x)
        x = self.act(x)
        # x = self.pre_logits(x)
        x = self.fc2(x)
        
        return x

# class SimilarityClassifier(nn.Module):
#     def __init__(self,
#                  conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()) -> None:
#         super().__init__()
        
        
#         self.norm = nn.LayerNorm(32*32)
        
#         self.fc = nn.Linear(64, 1)
#         self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(32*32, 64)),
#                 ('act', nn.Tanh()),
#                 # ('dropout', nn.Dropout(0.5))  # 添加一个 Dropout 层，丢弃概率为 0.5 第一版好像没用 dropout, 但是昨晚的测试有dropout
#             ]))
        
#     def forward(self, x):
#         B, C, _ = x.size()
#         x = x.view(B, C * C)
#         x = self.norm(x)
#         x = self.pre_logits(x)
#         x = self.fc(x)
#         # 使用局部最大池化操作对变形后的 Gram 矩阵进行降维
#         # pooled_matrix = F.max_pool1d(reshaped_gram.unsqueeze(2), kernel_size=256).squeeze(2)
        
        
#         return x



# class SimilarityClassifier(nn.Module):
#     def __init__(self, input_size=49*49, hidden_size=128, dropout=0.1):
#         super(SimilarityClassifier, self).__init__()
        
#         self.norm = nn.LayerNorm(input_size)
        
#         self.preprocess = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(input_size, hidden_size)),
#             ('act1', nn.ReLU()),
#             ('dropout1', nn.Dropout(dropout)),
#             ('fc2', nn.Linear(hidden_size, hidden_size)),
#             ('act2', nn.ReLU()),
#             ('dropout2', nn.Dropout(dropout))
#         ]))
        
#         self.fc = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.norm(x)
#         x = self.preprocess(x)
#         x = self.fc(x)
#         return x



# class SimilarityClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimilarityClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 12 * 12, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class HSIC(nn.Module):
    def __init__(self,
                 gamma = 5.) -> None:
        super().__init__()
        self.gamma = gamma

    @staticmethod
    def rbf_kernel(X, sigma=None):
        """Compute the RBF (Gaussian) kernel matrix of X using PyTorch."""
        GX = X @ X.T
        KX_diag = torch.diag(GX).unsqueeze(1)
        H = KX_diag + KX_diag.T - 2 * GX
        if sigma is None:
            # Using the median of the nonzero elements of H as the heuristic for sigma
            sigma = torch.median(H[H != 0])
        KX = torch.exp(-H / (2 * sigma ** 2))
        return KX

    @staticmethod
    def center_kernel(K):
        """Center a kernel matrix K in PyTorch, i.e., remove the mean in the feature space."""
        n = K.size(0)
        one_n = torch.ones(n, n, device=K.device) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        return K_centered

    
    def hsic(self, X, Y, sigma_x=None, sigma_y=None):
        """
        Calculate the HSIC statistic in PyTorch.
        
        X, Y are the input matrices of dimensions (n_samples, n_features),
        where each row is a sample. sigma_x and sigma_y are the bandwidths
        for the RBF kernel of X and Y, respectively. If None, the median
        heuristic is used to set the bandwidth.
        """
        KX = self.rbf_kernel(X, sigma_x)
        KY = self.rbf_kernel(Y, sigma_y)
        
        KX_centered = self.center_kernel(KX)
        KY_centered = self.center_kernel(KY)
        
        n = X.size(0)
        hsic_val = torch.trace(KX_centered @ KY_centered) / (n - 1) ** 2
        return hsic_val
    
    def forward(self, x, y):
        return self.gamma * self.hsic(x, y)


class MaxxVit(nn.Module):
    """ CoaTNet + MaxVit base model.

    Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            cfg: MaxxVitCfg,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
            padding=cfg.conv_cfg.padding,
            bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )
        stride = self.stem.stride
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        stages = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [MaxxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
                stage=i
            )]
            stride *= stage_stride
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        final_norm_layer = partial(get_norm_layer(cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
        self.head_hidden_size = cfg.head_hidden_size
        if self.head_hidden_size:
            self.norm = nn.Identity()
            self.head = patchHead(
                self.num_features,
                num_classes,
                hidden_size=self.head_hidden_size,
                pool_type=global_pool,
                drop_rate=drop_rate,
                norm_layer=final_norm_layer,
            )
            self.sim_head = SimilarityClassifier()
            self.cov_head = SimilarityClassifier()
            # self.head = agentClassifierHead()
            # self.total_head = agentClassifierHead(dim=768*3, hidden_dim=1024)
            # self.domain_head_real = patchHead(
            #     self.num_features,
            #     num_classes=2,
            #     hidden_size=self.head_hidden_size,
            #     pool_type=global_pool,
            #     drop_rate=drop_rate,
            #     norm_layer=final_norm_layer,
            # )
            # self.domain_head_fake = patchHead(
            #     self.num_features,
            #     num_classes=2,
            #     hidden_size=self.head_hidden_size,
            #     pool_type=global_pool,
            #     drop_rate=drop_rate,
            #     norm_layer=final_norm_layer,
            # )
        else:
            # standard classifier head w/ norm, pooling, fc classifier
            self.norm = final_norm_layer(self.num_features)
            self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # self.agent_norm0 = final_norm_layer(cfg.embed_dim[0])
        # self.agent_head0 = ClassifierHead(cfg.embed_dim[0], num_classes, pool_type=global_pool, drop_rate=drop_rate)
        # self.agent_norm1 = final_norm_layer(cfg.embed_dim[1])
        # self.agent_head1 = ClassifierHead(cfg.embed_dim[1], num_classes, pool_type=global_pool, drop_rate=drop_rate)
        # self.agent_norm2 = final_norm_layer(cfg.embed_dim[2])
        # self.agent_head2 = ClassifierHead(cfg.embed_dim[2], num_classes, pool_type=global_pool, drop_rate=drop_rate)
        # self.agent_norm3 = final_norm_layer(cfg.embed_dim[3])
        # self.agent_head3 = ClassifierHead(cfg.embed_dim[3], num_classes, pool_type=global_pool, drop_rate=drop_rate)

        self.hsic = HSIC()

        agent_num = 3
        self.agent = nn.Parameter(torch.zeros(1, agent_num, 64, 1, 1))
        nn.init.xavier_normal_(self.agent.data)  # 使用 Xavier/Glorot 初始化
        # nn.init.normal_(self.agent, std=1e-6)

        self.global_pos_row = nn.Parameter(torch.zeros(1, agent_num, 49))
        self.global_pos_col = nn.Parameter(torch.zeros(1, 49+agent_num, agent_num))
        
        self.gram_project = MbConvBlock(768, 147, cfg=cfg.conv_cfg)
                                  
        # self.disentangle = FeatureDisentangle(agent_num=agent_num)
        self.grl = GradientReversal()
        # self.agent_head = patchHead(dim=768, num_classes=num_classes)
        # self.patch_head = patchHead(dim=768, num_classes=num_classes)

        # self.domain_head = agentClassifierHead(dim=768, num_classes=4)
        
        self.tsne_fea = []
        self.tsne_label = []

        self.true_correct = 0
        self.true_total = 0
        self.fake_correct = 0
        self.fake_total = 0
        
        self.prob, self.label = [], []
        self.cov = []
        self.video_name = []
        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def erf_forward(self, x):
        x = self.stem(x)
        
        x, _, _ = self.stages((x, (self.agent, self.global_pos_row, self.global_pos_col), []))
        return x

    def forward_features(self, x):
        x = self.stem(x)
        
        x, agent, agentList = self.stages((x, (self.agent, self.global_pos_row, self.global_pos_col), []))
        x = self.norm(x)
        
        # logits = self.agent_head2(self.agent_norm2(torch.mean(agentList[2], dim=1)))
        # logits = logits+self.agent_head3(self.agent_norm3(torch.mean(agentList[3], dim=1)))
        # agent = torch.mean(agent, dim=1)
        # x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # print(x.shape, agent[0].shape)
        # print(agent.shape)
        # agent = torch.stack(agent, dim=0)
        return x, agent

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    @staticmethod
    def calculate_similarity(feature_map):
        """
        计算特征图的特征之间的相似度。
        
        参数:
            feature_map: 输入的特征图，维度为 (B, C, N, N)。
        
        返回:
            similarity_matrix: 特征之间的相似度矩阵，维度为 (B, N^2, N^2)。
        """
        B, C, N, _ = feature_map.shape
        # 将特征图重塑为 (B, C, N*N)，以便于计算
        feature_map = feature_map.view(B, C, N*N)
        # 归一化特征向量，以便于计算余弦相似度
        feature_map_norm = torch.nn.functional.normalize(feature_map, p=2, dim=1)
        
        # 计算特征之间的相似度，结果的维度将是 (B, N*N, N*N)
        similarity_matrix = torch.matmul(feature_map_norm.transpose(1, 2), feature_map_norm)
        
        return similarity_matrix
    
    
    def gram_matrix(self, feature_maps):
        """
        计算特征图的 Gram 矩阵
        参数:
            feature_maps: 形状为 (B, C, N, N) 的特征图
        返回:
            gram_matrix: Gram 矩阵，形状为 (B, C, C)
        """
        feature_maps = self.gram_project(feature_maps)
        # 获取批次大小 (B), 通道数 (C), 特征图高度 (N), 特征图宽度 (N)
        B, C, N, _ = feature_maps.size()

        # 将特征图重塑为形状为 (B, C, N*N) 的张量
        reshaped_feature_maps = feature_maps.view(B, C, N * N)

        # 计算每个特征图的 Gram 矩阵
        gram_matrix = torch.bmm(reshaped_feature_maps, reshaped_feature_maps.transpose(1, 2))

        # 归一化 Gram 矩阵
        gram_matrix = gram_matrix / (C * N * N)

        return gram_matrix
    
    def cov_matrix(self, feature_map):
        def normalize_covariance_matrix(covariance_matrices):
            # 计算对角线元素的标准差
            std_devs = torch.sqrt(torch.diagonal(covariance_matrices, dim1=1, dim2=2))
            # 标准化协方差矩阵
            normalized_covariance_matrices = covariance_matrices / (std_devs.unsqueeze(2) @ std_devs.unsqueeze(1))
            return normalized_covariance_matrices
        
        feature_map = self.gram_project(feature_map)
        B, C, N, _ = feature_map.size()
        # 将特征图展平为二维张量，形状为 (batch_size, channels, height * width)
        flattened_feature_map = feature_map.view(feature_map.size(0) * feature_map.size(1) // 49, 49, -1)

        # 计算每个通道的均值
        channel_means = torch.mean(flattened_feature_map, dim=2, keepdim=True)

        # 中心化特征图矩阵
        centered_feature_map = flattened_feature_map - channel_means

        # 将特征图转换为 (batch_size, channels, height*width) x (batch_size, height*width, channels)
        reshaped_feature_map = centered_feature_map.permute(0, 2, 1)

        # 计算协方差矩阵
        covariance_matrix = torch.matmul(reshaped_feature_map.transpose(1, 2), reshaped_feature_map) / (feature_map.size(2) * feature_map.size(3))
        # print(covariance_matrix)
        # covariance_matrix = normalize_covariance_matrix(covariance_matrix)
        
        return normalize_covariance_matrix(torch.mean(covariance_matrix.view(B, -1, 49, 49), dim=1))


    @staticmethod
    def CORAL(source, target):
        origin_point = torch.mean(source, 0, keepdim=True)# torch.tensor([0, 0], dtype=torch.float32).to(source.device)
        d = source.data.shape[1]
        Sigmoid = nn.Sigmoid()
        distances = torch.sum((source - origin_point)**2, dim=1).unsqueeze(-1)
        distances = distances / max(distances)
        distances = Sigmoid((distances-0.5)*10)
        # source covariance
        # xm = distances
        xm = (torch.mean(source, 0, keepdim=True) - source) * distances
        xc = xm.t() @ xm

        origin_point = torch.mean(target, 0, keepdim=True)
        distances = torch.sum((target - origin_point)**2, dim=1).unsqueeze(-1)
        distances = distances / max(distances)
        distances = Sigmoid((distances-0.5)*10)
        # target covariance
        # xmt = distances
        xmt = (torch.mean(target, 0, keepdim=True) - target) * distances
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        try:
            loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
            loss = loss/(4*d*d)
        except:
            print("error", xc.shape, xct.shape, target.shape, source.shape)
            return 0

        return loss

    def forward(self, x, inference=False):
        
        
        if inference == True:
            # x, y, video_name = x
            x, y = x
        # else:
        #     x, x_t = x
        
        # with torch.no_grad():
        x, agent = self.forward_features(x)
        B, C, H, W = x.shape
        


        
        logits, patch_logits = self.head(x)
        # kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        sigmoid = nn.Sigmoid()
        # logits_list = []
        # for i in range(3):
        #     # window: 
        #     i_logits = self.head(agent[:, i])
        #     # 015: i_logits = self.head(agent[:, i].detach())
        #     logits_list.append(i_logits)
            # print(logits_list[i].shape)
        
        # logits = self.head(self.calculate_similarity(x))
        
        # fea, logits, patch_logits = self.forward_head(x, pre_logits=True)
        # if inference == True:
        #     prediction_class = (logits.squeeze(-1) > 0.5)
        #     true_idx = (y == 0)
        #     correct = (prediction_class[true_idx] == y[true_idx]).sum().item()
        #     self.true_correct += correct
        #     self.true_total += y.shape[0] - (y.sum().item())
            
        #     correct = (prediction_class[~true_idx] == y[~true_idx]).sum().item()
        #     self.fake_correct += correct
        #     self.fake_total += y.sum().item()
        # return logits
        # return pred
        # logits = torch.softmax(pred, dim=1)[:, 1].contiguous()
        logits = sigmoid(logits).squeeze(-1)
        # # patch_logits = sigmoid(patch_logits.squeeze(-1)).view(B, -1)
        # for i in range(3):
        #     logits_list[i] = sigmoid(logits_list[i]).squeeze(-1)

        # cov_logits = self.cov_head(self.cov_matrix(x))

        if inference:
            # cov_logits = torch.softmax(cov_logits, dim=1)[:, 1].contiguous()
            try:
                torch.distributed.barrier()
                all_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                all_labels = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
                # all_covs = [torch.zeros_like(cov_logits) for _ in range(dist.get_world_size())]
                # all_video = [torch.zeros_like(video_name) for _ in range(dist.get_world_size())]
                
                dist.all_gather(all_labels, y)
                dist.all_gather(all_logits, logits)
                # dist.all_gather(all_covs, cov_logits)
                # dist.all_gather(all_video, video_name)

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                # all_covs = torch.cat(all_covs, dim=0)
                # all_video = torch.cat(all_video, dim=0)
                self.prob.append(
                    all_logits
                    .detach()
                    .squeeze()
                    .cpu()
                    # .numpy()
                )
                self.label.append(
                    all_labels
                    .detach()
                    .squeeze()
                    .cpu()
                    # .numpy()
                )
                # self.cov.append(
                #     all_covs
                #     .detach()
                #     .squeeze()
                #     .cpu()
                #     # .numpy()
                # )
            # self.video_name.append(
            #     all_video
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     # .numpy()
            # )
            except:
                self.prob.append(
                    logits
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                self.label.append(
                    y
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                # self.cov.append(
                #     all_covs
                #     .detach()
                #     .squeeze()
                #     .cpu()
                #     # .numpy()
                # )
            # self.tsne_fea.append(
            #     fea
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # prediction_class = (logits.squeeze(-1) > 0.5)
            # res = (prediction_class == y) * 4
            # label_tsne = y + label_spe + res
            # self.tsne_label.append(
            #     label_tsne
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # patch_logits = patch_logits.view(B, 7, 7).cpu().numpy()
            # for i in range(B):
            #     np.save(save_path[i], patch_logits[i])
            # self.video_name.append(
            #     video_name
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            return logits

        # softmax = nn.Softmax(dim=1)
        # kl_loss = kl_loss_fn(softmax(raw_logits.detach()/1.5).log(), softmax(logits/1.5))

        # hsic_loss = 0
        # for i in range(2):
        #     j = i+1
        #     while(j<3):
        #         hsic_loss = hsic_loss + self.hsic(agent[:, i].squeeze(), agent[:, j].squeeze())
        #         j = j+1

        # hsic_loss = self.hsic(self.cov_head(self.cov_matrix(x[:B // 2])), self.cov_head(self.cov_matrix(x[:B // 2])))
        # hsic_loss = hsic_loss + self.hsic(self.sim_head(self.calculate_similarity(x[:B // 2])), self.sim_head(self.calculate_similarity(x[B // 2:])))
        # hsic_loss = self.cov_matrix(x[:B // 2]), self.cov_head(self.cov_matrix(x[:B // 2])
        # hsic_loss = hsic_loss + self.hsic(self.sim_head(self.calculate_similarity(x[:B // 2])), self.sim_head(self.calculate_similarity(x[B // 2:])))
        
        # sim_logits = self.sim_head(self.calculate_similarity(x.detach()))
        
        # target_fea, _ = self.forward_features(x_t)
        # target_pred, target_patch = self.head(target_fea)
        # coral_loss = self.CORAL(patch_logits.view(B, -1), target_patch.view(B, -1))

        # patch_logits = sigmoid(patch_logits.squeeze(-1)).view(B, -1)
        return logits
        


def _create_maxxvit(variant, cfg_variant=None, pretrained=False, **kwargs):
    if cfg_variant is None:
        if variant in model_cfgs:
            cfg_variant = variant
        else:
            cfg_variant = '_'.join(variant.split('_')[:-1])
    return build_model_with_cfg(
        MaxxVit, variant, pretrained,
        model_cfg=model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)


@register_model
def multi_agentViT_base_tf_224(pretrained=False, **kwargs) -> MaxxVit:
    return _create_maxxvit('maxvit_base_tf_224', 'maxvit_base_tf', pretrained=pretrained, **kwargs)

@register_model
def multi_agentViT_small_tf_224(pretrained=False, **kwargs) -> MaxxVit:
    return _create_maxxvit('maxvit_small_tf_224', 'maxvit_small_tf', pretrained=pretrained, **kwargs)






























def _tf_cfg():
    return dict(
        conv_cfg=MaxxVitConvCfg(
            norm_eps=1e-3,
            act_layer='gelu',
            padding='same',
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            norm_eps=1e-5,
            act_layer='gelu',
            head_first=False,  # heads are interleaved (q_nh, q_hdim, k_nh, q_hdim, ....)
            rel_pos_type='bias_tf',
        ),
    )

def _rw_coat_cfg(
        stride_mode='pool',
        pool_type='avg2',
        conv_output_bias=False,
        conv_attn_early=False,
        conv_attn_act_layer='relu',
        conv_norm_layer='',
        transformer_shortcut_bias=True,
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        init_values=None,
        rel_pos_type='bias',
        rel_pos_dim=512,
):
    # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
    # Common differences for initial timm models:
    # - pre-norm layer in MZBConv included an activation after norm
    # - mbconv expansion calculated from input instead of output chs
    # - mbconv shortcut and final 1x1 conv did not have a bias
    # - SE act layer was relu, not silu
    # - mbconv uses silu in timm, not gelu
    # - expansion in attention block done via output proj, not input proj
    # Variable differences (evolved over training initial models):
    # - avg pool with kernel_size=2 favoured downsampling (instead of maxpool for coat)
    # - SE attention was between conv2 and norm/act
    # - default to avg pool for mbconv downsample instead of 1x1 or dw conv
    # - transformer block shortcut has no bias
    return dict(
        conv_cfg=MaxxVitConvCfg(
            stride_mode=stride_mode,
            pool_type=pool_type,
            pre_norm_act=True,
            expand_output=False,
            output_bias=conv_output_bias,
            attn_early=conv_attn_early,
            attn_act_layer=conv_attn_act_layer,
            act_layer='silu',
            norm_layer=conv_norm_layer,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            shortcut_bias=transformer_shortcut_bias,
            pool_type=pool_type,
            init_values=init_values,
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )


def _rw_max_cfg(
        stride_mode='dw',
        pool_type='avg2',
        conv_output_bias=False,
        conv_attn_ratio=1 / 16,
        conv_norm_layer='',
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        window_size=None,
        dim_head=32,
        init_values=None,
        rel_pos_type='bias',
        rel_pos_dim=512,
):
    # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
    # Differences of initial timm models:
    # - mbconv expansion calculated from input instead of output chs
    # - mbconv shortcut and final 1x1 conv did not have a bias
    # - mbconv uses silu in timm, not gelu
    # - expansion in attention block done via output proj, not input proj
    return dict(
        conv_cfg=MaxxVitConvCfg(
            stride_mode=stride_mode,
            pool_type=pool_type,
            expand_output=False,
            output_bias=conv_output_bias,
            attn_ratio=conv_attn_ratio,
            act_layer='silu',
            norm_layer=conv_norm_layer,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            pool_type=pool_type,
            dim_head=dim_head,
            window_size=window_size,
            init_values=init_values,
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )


def _next_cfg(
        stride_mode='dw',
        pool_type='avg2',
        conv_norm_layer='layernorm2d',
        conv_norm_layer_cl='layernorm',
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        window_size=None,
        no_block_attn=False,
        init_values=1e-6,
        rel_pos_type='mlp',  # MLP by default for maxxvit
        rel_pos_dim=512,
):
    # For experimental models with convnext instead of mbconv
    init_values = to_2tuple(init_values)
    return dict(
        conv_cfg=MaxxVitConvCfg(
            block_type='convnext',
            stride_mode=stride_mode,
            pool_type=pool_type,
            expand_output=False,
            init_values=init_values[0],
            norm_layer=conv_norm_layer,
            norm_layer_cl=conv_norm_layer_cl,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            pool_type=pool_type,
            window_size=window_size,
            no_block_attn=no_block_attn,  # enabled for MaxxViT-V2
            init_values=init_values[1],
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )

model_cfgs = dict(
    # timm specific CoAtNet configs
    coatnet_pico_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 3, 5, 2),
        stem_width=(32, 64),
        **_rw_max_cfg(  # using newer max defaults here
            conv_output_bias=True,
            conv_attn_ratio=0.25,
        ),
    ),
    coatnet_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(3, 4, 6, 3),
        stem_width=(32, 64),
        **_rw_max_cfg(  # using newer max defaults here
            stride_mode='pool',
            conv_output_bias=True,
            conv_attn_ratio=0.25,
        ),
    ),
    coatnet_0_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 3, 7, 2),  # deeper than paper '0' model
        stem_width=(32, 64),
        **_rw_coat_cfg(
            conv_attn_early=True,
            transformer_shortcut_bias=False,
        ),
    ),
    coatnet_1_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        stem_width=(32, 64),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_early=True,
            transformer_shortcut_bias=False,
        )
    ),
    coatnet_2_rw=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        stem_width=(64, 128),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_act_layer='silu',
            #init_values=1e-6,
        ),
    ),
    coatnet_3_rw=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        stem_width=(96, 192),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_act_layer='silu',
            init_values=1e-6,
        ),
    ),

    # Experimental CoAtNet configs w/ ImageNet-1k train (different norm layers, MLP rel-pos)
    coatnet_bn_0_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 3, 7, 2),  # deeper than paper '0' model
        stem_width=(32, 64),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_early=True,
            transformer_shortcut_bias=False,
            transformer_norm_layer='batchnorm2d',
        )
    ),
    coatnet_rmlp_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(3, 4, 6, 3),
        stem_width=(32, 64),
        **_rw_max_cfg(
            conv_output_bias=True,
            conv_attn_ratio=0.25,
            rel_pos_type='mlp',
            rel_pos_dim=384,
        ),
    ),
    coatnet_rmlp_0_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 3, 7, 2),  # deeper than paper '0' model
        stem_width=(32, 64),
        **_rw_coat_cfg(
            stride_mode='dw',
            rel_pos_type='mlp',
        ),
    ),
    coatnet_rmlp_1_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        stem_width=(32, 64),
        **_rw_coat_cfg(
            pool_type='max',
            conv_attn_early=True,
            transformer_shortcut_bias=False,
            rel_pos_type='mlp',
            rel_pos_dim=384,  # was supposed to be 512, woops
        ),
    ),
    coatnet_rmlp_1_rw2=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        stem_width=(32, 64),
        **_rw_coat_cfg(
            stride_mode='dw',
            rel_pos_type='mlp',
            rel_pos_dim=512,  # was supposed to be 512, woops
        ),
    ),
    coatnet_rmlp_2_rw=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        stem_width=(64, 128),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_act_layer='silu',
            init_values=1e-6,
            rel_pos_type='mlp'
        ),
    ),
    coatnet_rmlp_3_rw=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        stem_width=(96, 192),
        **_rw_coat_cfg(
            stride_mode='dw',
            conv_attn_act_layer='silu',
            init_values=1e-6,
            rel_pos_type='mlp'
        ),
    ),

    coatnet_nano_cc=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(3, 4, 6, 3),
        stem_width=(32, 64),
        block_type=('C', 'C', ('C', 'T'), ('C', 'T')),
        **_rw_coat_cfg(),
    ),
    coatnext_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(3, 4, 6, 3),
        stem_width=(32, 64),
        weight_init='normal',
        **_next_cfg(
            rel_pos_type='bias',
            init_values=(1e-5, None)
        ),
    ),

    # Trying to be like the CoAtNet paper configs
    coatnet_0=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 3, 5, 2),
        stem_width=64,
        head_hidden_size=768,
    ),
    coatnet_1=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        stem_width=64,
        head_hidden_size=768,
    ),
    coatnet_2=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        stem_width=128,
        head_hidden_size=1024,
    ),
    coatnet_3=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        stem_width=192,
        head_hidden_size=1536,
    ),
    coatnet_4=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 12, 28, 2),
        stem_width=192,
        head_hidden_size=1536,
    ),
    coatnet_5=MaxxVitCfg(
        embed_dim=(256, 512, 1280, 2048),
        depths=(2, 12, 28, 2),
        stem_width=192,
        head_hidden_size=2048,
    ),

    # Experimental MaxVit configs
    maxvit_pico_rw=MaxxVitCfg(
        embed_dim=(32, 64, 128, 256),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(24, 32),
        **_rw_max_cfg(),
    ),
    maxvit_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(1, 2, 3, 1),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(),
    ),
    maxvit_tiny_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(),
    ),
    maxvit_tiny_pm=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('PM',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(),
    ),

    maxvit_rmlp_pico_rw=MaxxVitCfg(
        embed_dim=(32, 64, 128, 256),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(24, 32),
        **_rw_max_cfg(rel_pos_type='mlp'),
    ),
    maxvit_rmlp_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(1, 2, 3, 1),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(rel_pos_type='mlp'),
    ),
    maxvit_rmlp_tiny_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(rel_pos_type='mlp'),
    ),
    maxvit_rmlp_small_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_rw_max_cfg(
            rel_pos_type='mlp',
            init_values=1e-6,
        ),
    ),
    maxvit_rmlp_base_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        head_hidden_size=768,
        **_rw_max_cfg(
            rel_pos_type='mlp',
        ),
    ),

    maxxvit_rmlp_nano_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(1, 2, 3, 1),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        weight_init='normal',
        **_next_cfg(),
    ),
    maxxvit_rmlp_tiny_rw=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(32, 64),
        **_next_cfg(),
    ),
    maxxvit_rmlp_small_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=(48, 96),
        **_next_cfg(),
    ),

    maxxvitv2_nano_rw=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(1, 2, 3, 1),
        block_type=('M',) * 4,
        stem_width=(48, 96),
        weight_init='normal',
        **_next_cfg(
            no_block_attn=True,
            rel_pos_type='bias',
        ),
    ),
    maxxvitv2_rmlp_base_rw=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 12, 2),
        block_type=('M',) * 4,
        stem_width=(64, 128),
        **_next_cfg(
            no_block_attn=True,
        ),
    ),
    maxxvitv2_rmlp_large_rw=MaxxVitCfg(
        embed_dim=(160, 320, 640, 1280),
        depths=(2, 6, 16, 2),
        block_type=('M',) * 4,
        stem_width=(80, 160),
        head_hidden_size=1280,
        **_next_cfg(
            no_block_attn=True,
        ),
    ),

    # Trying to be like the MaxViT paper configs
    maxvit_tiny_tf=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=512,
        **_tf_cfg(),
    ),
    maxvit_small_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_base_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 8, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_large_tf=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=128,
        stem_bias=True,
        head_hidden_size=1024,
        **_tf_cfg(),
    ),
    maxvit_xlarge_tf=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=192,
        stem_bias=True,
        head_hidden_size=1536,
        **_tf_cfg(),
    ),
)


def checkpoint_filter_fn(state_dict, model: nn.Module):
    model_state_dict = model.state_dict()
    out_dict = {}
    for k, v in state_dict.items():
        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )

        if k in model_state_dict and v.ndim != model_state_dict[k].ndim and v.numel() == model_state_dict[k].numel():
            # adapt between conv2d / linear layers
            assert v.ndim in (2, 4)
            v = v.reshape(model_state_dict[k].shape)
        out_dict[k] = v
    return out_dict
