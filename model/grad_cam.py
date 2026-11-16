""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

from sklearn import metrics
from metrics.base_metrics_class import calculate_metrics_for_train
from collections import OrderedDict
import numpy as np
import torch.distributed as dist
from torch import nn, optim, autograd

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


raw_image = None
USE_VISUALIZE = False

RETURN_ATTN = False

import matplotlib.pyplot as plt

def visualize_attn(attn, idx_layer):
    if not hasattr(visualize_attn, "plot"):
        visualize_attn.plot = plt.figure(figsize=(20,20))
        plt.subplot(13, 13, 6*13+1)
        plt.imshow(raw_image, cmap='inferno')
        plt.title('image')
    attn = attn.clone()
    num_heads = attn.size(1)
    attn = attn[0, :, 0, 1:].reshape(num_heads, -1)
    attn = attn.reshape(num_heads, int(224/16), int(224/16)).detach()
    attn = nn.functional.interpolate(attn.unsqueeze(
        0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    print(attn.shape)
    mean_attn = np.mean(attn, 0)
    plt.subplot(13, 13, idx_layer+1)
    plt.imshow(mean_attn)
    plt.title('Layer {} mean'.format(idx_layer))
    for i in range(num_heads):
        plt.subplot(13, 13, idx_layer + 1 + (i+1) * 13)
        plt.imshow(attn[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    if (idx_layer == 11):
        plt.tight_layout()
        plt.savefig("saved_attn_dino.png")
    return


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            idx_layer: int = 0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.idx_layer = idx_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        if USE_VISUALIZE: visualize_attn(attn, self.idx_layer)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.idx_layer == 11: return x, torch.mean(attn, dim=1)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            idx_layer: int = -1,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            idx_layer=idx_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.idx_layer = idx_layer
        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.idx_layer == 11: xt, attn = self.attn(self.norm1(x))
        else: xt = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(xt))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        # if self.idx_layer == 11: return x, attn
        return x


class ResPostBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """ Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_parallel: int = 2,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', mlp_layer(
                    dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_features, style_features, alpha=1):
        # print(content_features.shape)
        # 计算内容特征的均值和标准差
        content_mean, content_std = torch.mean(content_features, dim=1, keepdim=True), torch.std(content_features, dim=1, keepdim=True)
        
        # 计算风格特征的均值和标准差
        style_mean, style_std = torch.mean(style_features, dim=1, keepdim=True), torch.std(style_features, dim=1, keepdim=True)
        
        # 对内容特征进行归一化
        normalized_content = (content_features - content_mean) / (content_std + 1e-6)
        
        # 对归一化的内容特征进行风格化
        stylized_features = style_std * normalized_content + style_mean
        
        # 使用参数alpha控制风格化程度
        stylized_features = alpha * stylized_features + (1 - alpha) * content_features
        
        return stylized_features

import random


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
    def __init__(self, alpha=1):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, real, fake):
        # Normalize input features
        real = F.normalize(real, p=2, dim=1)
        fake = F.normalize(fake, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(real, fake.t())
        
        loss = torch.mean(similarity_matrix)

        return loss

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        self.low_patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        # self.low_patch_embed.apply(self.zero_initialize)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                idx_layer=i,
                # level=i
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        # self.bottle = nn.Linear(self.embed_dim, 64)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(3)])

        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, 2) # decoder to patch



        self.env_cls0 = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(1)])
        self.env_cls1 = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(1)])
        self.env_cls2 = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(1)])
        self.env_cls3 = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(1)])
        self.env_cls4 = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(1)])

        self.custom_init_weights(self.env_cls0[0])
        self.env_cls1.load_state_dict(self.env_cls0.state_dict())
        self.env_cls2.load_state_dict(self.env_cls0.state_dict())
        self.env_cls3.load_state_dict(self.env_cls0.state_dict())
        self.env_cls4.load_state_dict(self.env_cls0.state_dict())

        self.grl = GradientReversal()
        
        self.domain_head = nn.Linear(self.embed_dim, 4)
        
        self.tsne_fea = []
        self.tsne_label = []

        self.prob, self.label = [], []
        self.video_name = []

        # if weight_init != 'skip':
        #     self.init_weights(weight_init)

    @staticmethod
    def zero_initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Identity):
            pass  # nn.Identity 没有参数
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)


    @staticmethod
    def custom_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


    def visualize_attention(self, raw):
        globals()['raw_image'] = raw   

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x_m = x[:, 1:].contiguous().view(-1, 384)
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        
        return self.head(x), None, x if pre_logits else self.head(x)

    @staticmethod
    def CORAL(source, target):
        d = source.data.shape[1]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss/(4*d*d)

        return loss

    @staticmethod
    def inverse_CE(x, label, smoothing=0, temperature=1.):
        prob = F.softmax(x, dim=1)
        # print(label)
        axis = 1. / 4.            
        # print("before",prob)
        prob = torch.where(prob > axis, prob, 2. * axis - prob)
        prob = -prob + 1. + axis
        # print("after",prob)
        # prob = torch.clamp(prob, 0, 1)
        criterion = nn.NLLLoss()
        # 注意：NLLLoss 的输入是 log-probabilities，因此需要取对数

        log_softmax_output = torch.log(prob)

        # loss = -1 * warm * criterion(log_softmax_output, label) + 7.29
        loss = -temperature * criterion(log_softmax_output, label)
        return loss

    def forward_decoder(self, x):
        # embed tokens
        # x = self.decoder_embed(x)


        # # add pos embed
        # x = x + self.decoder_pos_embed

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)

        # predictor projection
        x = self.head(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    @staticmethod
    def generate_mask(attention_map, percentage=10):
        # 获取attention_map的形状
        B, N = attention_map.shape
        
        # 计算需要mask的数量
        k = int(N * percentage / 100)
        
        # 初始化mask，全部为1
        mask = np.ones((B, N), dtype=int)
        
        # 找到每行中前k大的索引
        idx = np.argsort(attention_map.detach().cpu(), axis=1)[:, -k:]
        # 使用高级索引将这些索引对应的位置置为False
        row_indices = np.arange(B)[:, np.newaxis]
        mask[row_indices, idx] = 0
        # idx = np.argsort(attention_map.detach().cpu(), axis=1)[:, :k]
        # mask[row_indices, idx] = 0
        return mask

    def forward(self, x: torch.Tensor, inference=False, GradCAM=False, step_cnt=0) -> torch.Tensor:
        if inference == True:
            if isinstance(x, tuple):
                x, y = x
        else:
            if isinstance(x, tuple):
                x, compress = x
            
        # LQ_idx = (compress > 0.5)
        # if any(LQ_idx):
        #     x_raw = self.patch_embed(x)
        #     x_raw[LQ_idx] = self.low_patch_embed(x[LQ_idx])
        #     x = x_raw
        # else: 
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        # if inference == False:
        #     x_d, attn = self.forward_features(x)
        #     mask = self.generate_mask(attn)
        #     x1 = torch.cat((x[:,0:1], x[:, 1:]*torch.tensor(mask, device=x.device).unsqueeze(-1)), dim=1)
        #     x1d, attn = self.forward_features(x1)
        #     mask = self.generate_mask(attn)
        #     x2 = torch.cat((x[:,0:1], x1[:, 1:]*torch.tensor(mask, device=x.device).unsqueeze(-1)), dim=1)
        #     x2d, attn = self.forward_features(x2)
        
        #     x = torch.cat((x_d, x1d, x2d))
        # else:
        x = self.forward_features(x)
        Softmax = nn.Softmax(dim=-1)

        pred, _, fea = self.forward_head(x, pre_logits=True)
        
        env_loss = 0
        inv_loss = 0
        neg_loss = 0
        env_pred = None
        loss_fn = nn.CrossEntropyLoss()

        if inference:
            logits = Softmax(pred)[:, 1].contiguous()
            try:
                torch.distributed.barrier()
                all_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                all_labels = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
                # all_video = [torch.zeros_like(video_name) for _ in range(dist.get_world_size())]

                dist.all_gather(all_labels, y)
                dist.all_gather(all_logits, logits)
                # dist.all_gather(all_video, video_name)

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
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
            # self.video_name.append(
            #     all_video
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     # .numpy()
            # )
            except:
                try:
                    self.prob.append(
                        logits
                        .detach()
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    self.label.append(
                        y
                        .detach()
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                except:
                    pass
            # self.video_name.append(
            #     video_name
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            return logits

        # dec_x = self.forward_decoder(x).reshape(-1, 2)

        return pred, None, env_loss, env_pred, inv_loss


    def get_train_metrics(self, data_dict: dict, pred) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        if isinstance(pred, tuple):
            pred = pred[0]
        # get pred and label
        label = data_dict['label']
        # label_spe = data_dict['label_spe']
        # pred_spe = pred_dict['cls_spe']
        # compute metrics for batch data
        # print(label.shape, pred.shape)
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        def get_video_auc(f_label_list, v_name_list, f_pred_list):
            video_res_dict = dict()
            video_pred_list = list()
            video_label_list = list()
            # summarize all the results for each video
            for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
                if video not in video_res_dict.keys():
                    video_res_dict[video] = {"scores": [score], "label": label}
                else:
                    video_res_dict[video]["scores"].append(score)
            # get the score and label for each video
            for video, res in video_res_dict.items():
                score = sum(res['scores']) / len(res['scores'])
                label = res['label']
                video_pred_list.append(score)
                video_label_list.append(label)

            v_auc = metrics.roc_auc_score(video_label_list, video_pred_list)
            return v_auc

        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # print(y_pred.shape, y_true.shape)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        
        acc = 0
        
        # video_auc = get_video_auc(y_true, np.concatenate(self.video_name), y_pred)

        # self.correct, self.total = 0, 0

        # reset the prob and label
        self.prob, self.label = [], []
        self.video_name = []
        return {#'video_auc':video_auc, 
                'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}


    @staticmethod
    def calc_loss(pred, labels, step_cnt=0):
        def mixup_criterion(outputs, targets_a, targets_b, mix_weights):
            loss_a = F.cross_entropy(outputs, targets_a, reduction='none')
            loss_b = F.cross_entropy(outputs, targets_b, reduction='none')
            mixed_loss = mix_weights * loss_a + (1 - mix_weights) * loss_b
            return torch.mean(mixed_loss)
        def penalty(logits, y):
            scale = torch.tensor(1.).cuda().requires_grad_()
            loss_ce = torch.nn.CrossEntropyLoss()
            loss = loss_ce(logits * scale, y)
            grad = autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad**2)
        # cos_fn = CosineDistanceLoss()
        # real_pred = pred[0][(labels==0)]
        # fake_pred = pred[0][(labels!=0)]
        # if (labels==0).any() and (labels!=0).any():
        #     cos_loss = cos_fn(real_pred, fake_pred) + 1
        # else: cos_loss = 0
        # targets_a, targets_b, mix_weights = labels
        # y0, y1, y2 = torch.chunk(labels, 3, dim=0)
        loss_fn = nn.CrossEntropyLoss()
        # ERM_label = torch.cat((y1, y2), dim=0)
        loss_cls = loss_fn(pred[0], labels)
        # return {'overall': loss_cls}
        # cos_weight = 1.0 if step_cnt > 1000 else 0.0
        # loss_balance = torch.abs(torch.mean(pred[0]/torch.abs(pred[0]), dim=1)).mean()
        return {'overall': loss_cls, 'cls':loss_cls}
        softmax = nn.Softmax(dim=-1)
        loss_patch = 0
        for i in range(pred[1].shape[0]):
            patch_logits = pred[1][i]
            label_i = torch.zeros((196,)).to(patch_logits.device)
            logits = softmax(patch_logits)[:, 1]
            greater_idx = (logits > 0.5)
            count_greater_than_0_5 = greater_idx.sum().item()
            if labels[i] == 0: # real
                loss_patch = loss_patch + loss_fn(patch_logits, label_i.long())
            else: 
                if count_greater_than_0_5 >= 6:   #10
                    label_i[greater_idx] = 1
                    fake_logits = patch_logits[greater_idx]
                    fake_label = label_i[greater_idx]
                    real_logits = patch_logits[~greater_idx]
                    real_label = label_i[~greater_idx]
                    fake_loss = loss_fn(fake_logits, fake_label.long())
                    if torch.numel(real_label) != 0:
                        real_loss = loss_fn(real_logits, real_label.long())
                    else: real_loss = 0
                    # if torch.isnan(real_loss):

                    #     print("real loss", real_loss, '\n', patch_logits, '\n', real_logits, '\n', real_label)
                    # 0.7 0.3
                    loss_patch = loss_patch + 0.3 * fake_loss + real_loss * 0.2
                else:
                    label_i = torch.zeros((196,)).to(patch_logits.device)
                    loss_patch = loss_patch + 0.08 * loss_fn(patch_logits, label_i.long())
            loss_patch = loss_patch / 196.
            # print("Batch", i, patch_logits)
            # print(patch_logits)
            # print()
        # real = fea[(labels == 0)]
        # fake = fea[(labels == 1)]
        # loss_con = calc_contrastive_loss(real, fake)
        
        # loss_ce_fn = nn.CrossEntropyLoss()
        # real_idx = (labels == 0)
        # fake_idx = (labels == 1)
        # print("True", labels[real_idx], label_spe[real_idx], domain_pred[real_idx])
        # loss_inv = loss_ce_fn(domain_pred[0][real_idx], label_spe[real_idx])+loss_ce_fn(domain_pred[1][fake_idx], label_spe[fake_idx])

        return {'overall': 0.95*loss_cls + loss_patch*0.05, 
                'cls':loss_cls,
                'patch':loss_patch,
                }

        # IRM_label = torch.cat((y0, y2), dim=0)
        # IRM_loss = penalty(pred[2], IRM_label)
        # loss = mixup_criterion(pred, targets_a, targets_b, mix_weights)
        # if IRM_loss < 1e-2: IRM_loss = IRM_loss * 1e2
        return {'overall':loss_cls,
                }
        
    @staticmethod
    def calc_test_loss(pred, labels):
        loss_fn = nn.BCELoss()
        return {'overall':loss_fn(pred, labels.float())}

    @staticmethod
    def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            """计算Gram核矩阵
            source: sample_size_1 * feature_size 的数据
            target: sample_size_2 * feature_size 的数据
            kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
            kernel_num: 表示的是多核的数量
            fix_sigma: 表示是否使用固定的标准差
                return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                                矩阵，表达形式:
                                [   K_ss K_st
                                    K_ts K_tt ]
            """
            n_samples = int(source.size()[0])+int(target.size()[0])
            total = torch.cat([source, target], dim=0) # 合并在一起

            total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                            int(total.size(0)), \
                                            int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                            int(total.size(0)), \
                                            int(total.size(1)))
            L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

            # 计算多核中每个核的bandwidth
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

            # 高斯核的公式，exp(-|x-y|/bandwith)
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                        bandwidth_temp in bandwidth_list]

            return sum(kernel_val) # 将多个核合并在一起

        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n] 
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
            
        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


class NoiseCNN(nn.Module):
    def __init__(self):
        super(NoiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # Assuming input image size 224x224
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.fc2 = nn.Linear(128, 1)  # Output a single value for noise estimation

        self.prob, self.label = [], []

    def forward(self, x, inference=False):
        if inference == True:
            x, y = x
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 28 * 28)  # Reshape for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        sigmoid = nn.Sigmoid()
        logits = sigmoid(x).squeeze(-1)

        return logits

class RefineNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=2)
        self.backbone = _create_vision_transformer('vit_small_patch16_224', **dict(model_args))
        self.compression_level = NoiseCNN()

        self.prob, self.label = [], []
        self.video_name = []

    def forward(self, x, inference=False):
        if inference == True:
            x, y = x
        compress = self.compression_level(x)
        compress = torch.ones_like(compress)
        pred = self.backbone((x, compress))[0]
        
        Softmax = nn.Softmax(dim=-1)
        logits = Softmax(pred)[:, 1].contiguous()

        if inference:
            try:
                torch.distributed.barrier()
                all_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                all_labels = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
                # all_video = [torch.zeros_like(video_name) for _ in range(dist.get_world_size())]


                dist.all_gather(all_labels, y)
                dist.all_gather(all_logits, logits)
                # dist.all_gather(all_video, video_name)

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
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
            # self.video_name.append(
            #     all_video
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     # .numpy()
            # )
            except:
                try:
                    self.prob.append(
                        logits
                        .detach()
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    self.label.append(
                        y
                        .detach()
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                except:
                    pass
            # self.video_name.append(
            #     video_name
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            return logits
        return pred
    
    def get_train_metrics(self, data_dict: dict, pred) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        if isinstance(pred, tuple):
            pred = pred[0]
        # get pred and label
        label = data_dict['label']
        # label_spe = data_dict['label_spe']
        # pred_spe = pred_dict['cls_spe']
        # compute metrics for batch data
        # print(label.shape, pred.shape)
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        def get_video_auc(f_label_list, v_name_list, f_pred_list):
            video_res_dict = dict()
            video_pred_list = list()
            video_label_list = list()
            # summarize all the results for each video
            for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
                if video not in video_res_dict.keys():
                    video_res_dict[video] = {"scores": [score], "label": label}
                else:
                    video_res_dict[video]["scores"].append(score)
            # get the score and label for each video
            for video, res in video_res_dict.items():
                score = sum(res['scores']) / len(res['scores'])
                label = res['label']
                video_pred_list.append(score)
                video_label_list.append(label)

            v_auc = metrics.roc_auc_score(video_label_list, video_pred_list)
            return v_auc

        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # print(y_pred.shape, y_true.shape)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        
        acc = 0
        
        # video_auc = get_video_auc(y_true, np.concatenate(self.video_name), y_pred)

        # self.correct, self.total = 0, 0

        # reset the prob and label
        self.prob, self.label = [], []
        self.video_name = []
        return {#'video_auc':video_auc, 
                'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}


    @staticmethod
    def calc_loss(pred, labels, step_cnt=0):
        def mixup_criterion(outputs, targets_a, targets_b, mix_weights):
            loss_a = F.cross_entropy(outputs, targets_a, reduction='none')
            loss_b = F.cross_entropy(outputs, targets_b, reduction='none')
            mixed_loss = mix_weights * loss_a + (1 - mix_weights) * loss_b
            return torch.mean(mixed_loss)
        def penalty(logits, y):
            scale = torch.tensor(1.).cuda().requires_grad_()
            loss_ce = torch.nn.CrossEntropyLoss()
            loss = loss_ce(logits * scale, y)
            grad = autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad**2)
        # cos_fn = CosineDistanceLoss()
        # real_pred = pred[0][(labels==0)]
        # fake_pred = pred[0][(labels!=0)]
        # if (labels==0).any() and (labels!=0).any():
        #     cos_loss = cos_fn(real_pred, fake_pred) + 1
        # else: cos_loss = 0
        # targets_a, targets_b, mix_weights = labels
        # y0, y1, y2 = torch.chunk(labels, 3, dim=0)
        loss_fn = nn.CrossEntropyLoss()
        # ERM_label = torch.cat((y1, y2), dim=0)
        loss_cls = loss_fn(pred[0], labels)
        # return {'overall': loss_cls}
        # cos_weight = 1.0 if step_cnt > 1000 else 0.0
        # loss_balance = torch.abs(torch.mean(pred[0]/torch.abs(pred[0]), dim=1)).mean()
        return {'overall': loss_cls, 'cls':loss_cls}
        softmax = nn.Softmax(dim=-1)
        loss_patch = 0
        for i in range(pred[1].shape[0]):
            patch_logits = pred[1][i]
            label_i = torch.zeros((196,)).to(patch_logits.device)
            logits = softmax(patch_logits)[:, 1]
            greater_idx = (logits > 0.5)
            count_greater_than_0_5 = greater_idx.sum().item()
            if labels[i] == 0: # real
                loss_patch = loss_patch + loss_fn(patch_logits, label_i.long())
            else: 
                if count_greater_than_0_5 >= 6:   #10
                    label_i[greater_idx] = 1
                    fake_logits = patch_logits[greater_idx]
                    fake_label = label_i[greater_idx]
                    real_logits = patch_logits[~greater_idx]
                    real_label = label_i[~greater_idx]
                    fake_loss = loss_fn(fake_logits, fake_label.long())
                    if torch.numel(real_label) != 0:
                        real_loss = loss_fn(real_logits, real_label.long())
                    else: real_loss = 0
                    # if torch.isnan(real_loss):

                    #     print("real loss", real_loss, '\n', patch_logits, '\n', real_logits, '\n', real_label)
                    # 0.7 0.3
                    loss_patch = loss_patch + 0.3 * fake_loss + real_loss * 0.2
                else:
                    label_i = torch.zeros((196,)).to(patch_logits.device)
                    loss_patch = loss_patch + 0.08 * loss_fn(patch_logits, label_i.long())
            loss_patch = loss_patch / 196.
            # print("Batch", i, patch_logits)
            # print(patch_logits)
            # print()
        # real = fea[(labels == 0)]
        # fake = fea[(labels == 1)]
        # loss_con = calc_contrastive_loss(real, fake)
        
        # loss_ce_fn = nn.CrossEntropyLoss()
        # real_idx = (labels == 0)
        # fake_idx = (labels == 1)
        # print("True", labels[real_idx], label_spe[real_idx], domain_pred[real_idx])
        # loss_inv = loss_ce_fn(domain_pred[0][real_idx], label_spe[real_idx])+loss_ce_fn(domain_pred[1][fake_idx], label_spe[fake_idx])

        return {'overall': 0.95*loss_cls + loss_patch*0.05, 
                'cls':loss_cls,
                'patch':loss_patch,
                }

        # IRM_label = torch.cat((y0, y2), dim=0)
        # IRM_loss = penalty(pred[2], IRM_label)
        # loss = mixup_criterion(pred, targets_a, targets_b, mix_weights)
        # if IRM_loss < 1e-2: IRM_loss = IRM_loss * 1e2
        return {'overall':loss_cls,
                }
        
    @staticmethod
    def calc_test_loss(pred, labels):
        loss_fn = nn.BCELoss()
        return {'overall':loss_fn(pred, labels.float())}

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=768, out_ch=128,dropout=0.2):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 384
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]
        # add_block += [nn.Dropout(p=dropout)]
        # add_block += [nn.Linear(num_bottleneck, 256)]
        # add_block += [nn.BatchNorm1d(256)]
        # add_block += [nn.ReLU()]
        add_block += [nn.Dropout(p=dropout)]
        add_block += [nn.Linear(num_bottleneck, out_ch)]
        add_block += [nn.Linear(out_ch, 2)]
        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x

class Dual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=2)

        self.specific = _create_vision_transformer('vit_small_patch16_224', **dict(model_args))
        self.invariant = _create_vision_transformer('vit_small_patch16_224', **dict(model_args))

        self.head = nn.Linear(384, 2)
        # self.head = ChannelCompress()

        self.prob, self.label = [], []
        self.video_name = []


    def forward(self, x: torch.Tensor, inference=False) -> torch.Tensor:
        if inference == True:
            x, y = x
        # x = self.batch_high_pass_filter(x)
        # print(x.shape)
        pred1, fea1, spe_pred = self.specific(x)
        with torch.no_grad():
            pred2, fea2, inv_pred = self.invariant(x)
        pred = self.head(fea1-fea2)
        # fea = torch.cat((spe_fea, inv_fea), dim=1)
        # print(fea.shape)
        # in_pred = self.in_head(fea)
        # pred = self.head(fea)
        # if inference == False:
        #     real_x_rand, real_x_source, fake_x_source = torch.chunk(x, 3, dim=0)
        #     train_x = torch.cat((real_x_source, fake_x_source))
        #     pred = self.forward_head(train_x)
        #     # IRM_x = torch.cat((real_x_rand, fake_x_source))
        #     # IRM_pred = self.forward_head(IRM_x)
        #     # total_pred = self.forward_head(x)
        # else:
        # pred = self.sim_head(self.calculate_similarity(x[:, 1:]))
        # pred, patch_pred, fea = self.forward_head(x, pre_logits=True)
        # domain_pred = self.domain_head(fea)
        # try:
        #     B = x.shape[0]
        #     idx_list = list(range(0, B//2))
            
        #     random.shuffle(idx_list)
        #     style_fake_x = self.AdaIN(fake_x, real_x[idx_list], alpha=0.7)
        #     random.shuffle(idx_list)
        #     style_real_x = self.AdaIN(real_x, fake_x[idx_list], alpha=0.7)

        #     criterion = nn.MSELoss()   
        #     l2_loss = criterion(style_fake_x, fake_x) + criterion(style_real_x, real_x)

        #     # if np.random.uniform() < 0.5:
        #     style_x = torch.cat((style_fake_x, style_real_x), dim=0)
        #     style_pred = self.forward_head(style_x)
        # except:
        # style_pred = None
        # l2_loss = 0
        # fea = fea.view(B, C, -1)
        # logits = sigmoid(self.k_means_distance(x))
        # print(x.shape)
        # patch_x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # patch_logits = sigmoid(self.forward_patch_head(patch_x)).view(B, -1)
        
        

        # logits = sigmoid(self.head(self.calculate_similarity(fea) + self.gram_matrix(fea)).squeeze(-1))
        
        if inference:
            softmax = nn.Softmax(dim=-1)
            logits = softmax(pred)[:, 1].contiguous()
            try:
                torch.distributed.barrier()
                all_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                all_labels = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
                # all_video = [torch.zeros_like(video_name) for _ in range(dist.get_world_size())]


                dist.all_gather(all_labels, y)
                dist.all_gather(all_logits, logits)
                # dist.all_gather(all_video, video_name)

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
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
            # self.video_name.append(
            #     video_name
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            return logits

        return pred, None#, in_pred, spe_pred, inv_pred, cls_pred1, cls_pred2


    def get_train_metrics(self, data_dict: dict, pred) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        if isinstance(pred, tuple):
            pred = pred[0]
        # get pred and label
        label = data_dict['label']
        # label_spe = data_dict['label_spe']
        # pred_spe = pred_dict['cls_spe']
        # compute metrics for batch data
        # print(label.shape, pred.shape)
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        def get_video_auc(f_label_list, v_name_list, f_pred_list):
            video_res_dict = dict()
            video_pred_list = list()
            video_label_list = list()
            # summarize all the results for each video
            for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
                if video not in video_res_dict.keys():
                    video_res_dict[video] = {"scores": [score], "label": label}
                else:
                    video_res_dict[video]["scores"].append(score)
            # get the score and label for each video
            for video, res in video_res_dict.items():
                score = sum(res['scores']) / len(res['scores'])
                label = res['label']
                video_pred_list.append(score)
                video_label_list.append(label)

            v_auc = metrics.roc_auc_score(video_label_list, video_pred_list)
            return v_auc

        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # print(y_pred.shape, y_true.shape)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        
        acc = 0
        
        # video_auc = get_video_auc(y_true, np.concatenate(self.video_name), y_pred)

        # self.correct, self.total = 0, 0

        # reset the prob and label
        self.prob, self.label = [], []
        self.video_name = []
        return {#'video_auc':video_auc, 
                'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}


    @staticmethod
    def calc_loss(pred, labels, step_cnt=0):
        def mixup_criterion(outputs, targets_a, targets_b, mix_weights):
            loss_a = F.cross_entropy(outputs, targets_a, reduction='none')
            loss_b = F.cross_entropy(outputs, targets_b, reduction='none')
            mixed_loss = mix_weights * loss_a + (1 - mix_weights) * loss_b
            return torch.mean(mixed_loss)
        def penalty(logits, y):
            scale = torch.tensor(1.).cuda().requires_grad_()
            loss_ce = torch.nn.CrossEntropyLoss()
            loss = loss_ce(logits * scale, y)
            grad = autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad**2)
        # targets_a, targets_b, mix_weights = labels
        # y0, y1, y2 = torch.chunk(labels, 3, dim=0)
        loss_fn = nn.CrossEntropyLoss()
        # ERM_label = torch.cat((y1, y2), dim=0)
        loss_cls = loss_fn(pred[0], labels)# + loss_fn(pred[1], labels)
        return {'overall': loss_cls, 'cls':loss_cls}
        softmax = nn.Softmax(dim=-1)
        loss_patch = 0
        for i in range(pred[1].shape[0]):
            patch_logits = pred[1][i]
            label_i = torch.zeros((196,)).to(patch_logits.device)
            logits = softmax(patch_logits)[:, 1]
            greater_idx = (logits > 0.5)
            count_greater_than_0_5 = greater_idx.sum().item()
            if labels[i] == 0: # real
                loss_patch = loss_patch + loss_fn(patch_logits, label_i.long())
            else:
                if count_greater_than_0_5 >= 6:   #10
                    label_i[greater_idx] = 1
                    fake_logits = patch_logits[greater_idx]
                    fake_label = label_i[greater_idx]
                    real_logits = patch_logits[~greater_idx]
                    real_label = label_i[~greater_idx]
                    fake_loss = loss_fn(fake_logits, fake_label.long())
                    if torch.numel(real_label) != 0:
                        real_loss = loss_fn(real_logits, real_label.long())
                    else: real_loss = 0
                    # if torch.isnan(real_loss):

                    #     print("real loss", real_loss, '\n', patch_logits, '\n', real_logits, '\n', real_label)
                    # 0.7 0.3
                    loss_patch = loss_patch + 0.3 * fake_loss + real_loss * 0.2
                else:
                    label_i = torch.zeros((196,)).to(patch_logits.device)
                    loss_patch = loss_patch + 0.08 * loss_fn(patch_logits, label_i.long())
            loss_patch = loss_patch / 196.
            # print("Batch", i, patch_logits)
            # print(patch_logits)
            # print()
        # real = fea[(labels == 0)]
        # fake = fea[(labels == 1)]
        # loss_con = calc_contrastive_loss(real, fake)
        
        # loss_ce_fn = nn.CrossEntropyLoss()
        # real_idx = (labels == 0)
        # fake_idx = (labels == 1)
        # print("True", labels[real_idx], label_spe[real_idx], domain_pred[real_idx])
        # loss_inv = loss_ce_fn(domain_pred[0][real_idx], label_spe[real_idx])+loss_ce_fn(domain_pred[1][fake_idx], label_spe[fake_idx])

        return {'overall': 0.95*loss_cls + loss_patch*0.05, 
                'cls':loss_cls,
                'patch':loss_patch,
                }

        # IRM_label = torch.cat((y0, y2), dim=0)
        # IRM_loss = penalty(pred[2], IRM_label)
        # loss = mixup_criterion(pred, targets_a, targets_b, mix_weights)
        # if IRM_loss < 1e-2: IRM_loss = IRM_loss * 1e2
        return {'overall':loss_cls,
                }
        
    @staticmethod
    def calc_test_loss(pred, labels):
        loss_fn = nn.BCELoss()
        return {'overall':loss_fn(pred, labels.float())}

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> None:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {

    # re-finetuned augreg 21k FT on in1k weights
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'vit_base_patch16_384.augreg2_in21k_ft_in1k': _cfg(),
    'vit_base_patch8_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),

    # How to train your ViT (augreg) weights, pretrained on 21k FT on in1k
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    'vit_base_patch16_224.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        hf_hub_id='timm/'),
    'vit_base_patch16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),

    # How to train your ViT (augreg) weights trained on in1k only
    'vit_small_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224.untrained': _cfg(url=''),
    'vit_huge_patch14_224.untrained': _cfg(url=''),
    'vit_giant_patch14_224.untrained': _cfg(url=''),
    'vit_gigantic_patch14_224.untrained': _cfg(url=''),

    # patch models, imagenet21k (weights from official Google JAX impl), classifier not valid
    'vit_base_patch32_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_base_patch16_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_large_patch32_224.orig_in21k': _cfg(
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_large_patch16_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_huge_patch14_224.orig_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=0),

    # How to train your ViT (augreg) weights, pretrained on in21k
    'vit_tiny_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_large_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz', custom_load=True,
        hf_hub_id='timm/'),
    'vit_base_patch16_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz', custom_load=True,
        hf_hub_id='timm/'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    # DINOv2 pretrained - https://arxiv.org/abs/2304.07193 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # DINOv2 pretrained w/ registers - https://arxiv.org/abs/2309.16588 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_in21k_miil-887286df.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221),
    'vit_base_patch16_224_miil.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_1k_miil_84_4-2deb18e3.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear'),

    # Custom timm variants
    'vit_base_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_base_patch16_rpn_224-sw-3b07e89d.pth',
        hf_hub_id='timm/'),
    'vit_medium_patch16_gap_240.sw_in12k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=11821),
    'vit_medium_patch16_gap_256.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95),
    'vit_medium_patch16_gap_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=0.95, crop_mode='squash'),
    'vit_base_patch16_gap_224': _cfg(),

    # CLIP pretrained image tower and related fine-tuned weights
    'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 384, 384)),
    'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 448, 448)),
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in12k_in1k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_base_patch16_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_base_patch16_clip_384.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_384.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),

    'vit_base_patch32_clip_224.laion2b_ft_in12k': _cfg(
        #hf_hub_id='timm/vit_base_patch32_clip_224.laion2b_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=11821),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.openai_ft_in12k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_giant_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_gigantic_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1280),

    'vit_base_patch32_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch32_clip_256.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 256, 256), num_classes=512),
    'vit_base_patch16_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),

    'vit_base_patch16_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-B-16',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-L-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_huge_patch14_clip_378.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14-378',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        crop_pct=1.0, input_size=(3, 378, 378), num_classes=1024),

    'vit_base_patch32_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-b32-fullcc2.5b',
        hf_hub_filename='metaclip_b32_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch16_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-b16-fullcc2.5b',
        hf_hub_filename='metaclip_b16_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-l14-fullcc2.5b',
        hf_hub_filename='metaclip_l14_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-h14-fullcc2.5b',
        hf_hub_filename='metaclip_h14_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),

    'vit_base_patch32_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_base_patch32_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_base_patch16_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_large_patch14_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_large_patch14_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_336.openai': _cfg(
        hf_hub_id='timm/vit_large_patch14_clip_336.openai', hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), num_classes=768),

    # experimental (may be removed)
    'vit_base_patch32_plus_256.untrained': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),
    'vit_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240), crop_pct=0.95),
    'vit_small_patch16_36x1_224.untrained': _cfg(url=''),
    'vit_small_patch16_18x2_224.untrained': _cfg(url=''),
    'vit_base_patch16_18x2_224.untrained': _cfg(url=''),

    # EVA fine-tuned weights from MAE style MIM - EVA-CLIP target pretrain
    # https://github.com/baaivision/EVA/blob/7ecf2c0a370d97967e86d047d7af9188f78d2df3/eva/README.md#eva-l-learning-better-mim-representations-from-eva-clip
    'eva_large_patch14_196.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_21k_to_1k_ft_88p6.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_21k_to_1k_ft_89p2.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
    'eva_large_patch14_196.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_1k_ft_88p0.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_1k_ft_88p65.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

    'flexivit_small.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.1000ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_1000ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.300ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'flexivit_large.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.patch16_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b16_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.patch30_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b30_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'vit_base_patch16_xp_224.untrained': _cfg(url=''),
    'vit_large_patch14_xp_224.untrained': _cfg(url=''),
    'vit_huge_patch14_xp_224.untrained': _cfg(url=''),

    'vit_base_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_large_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_huge_patch14_gap_224.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch16_gap_448.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        input_size=(3, 448, 448), crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_giant_patch16_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_base_patch16_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_base_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_base_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_base_patch16_siglip_512.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-512',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 512, 512),
        num_classes=0),
    'vit_large_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_large_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_so400m_patch14_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_so400m_patch14_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),

    'vit_medium_patch16_reg4_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_medium_patch16_reg4_gap_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_base_patch16_reg8_gap_256': _cfg(input_size=(3, 256, 256)),
}

_quick_gelu_cfgs = [
    'vit_large_patch14_clip_224.dfn2b',
    'vit_huge_patch14_clip_224.dfn5b',
    'vit_huge_patch14_clip_378.dfn5b',
    'vit_base_patch32_clip_224.metaclip_2pt5b',
    'vit_base_patch16_clip_224.metaclip_2pt5b',
    'vit_large_patch14_clip_224.metaclip_2pt5b',
    'vit_huge_patch14_clip_224.metaclip_2pt5b',
    'vit_base_patch32_clip_224.openai',
    'vit_base_patch16_clip_224.openai',
    'vit_large_patch14_clip_224.openai',
    'vit_large_patch14_clip_336.openai',
]
default_cfgs.update({
    n.replace('_clip_', '_clip_quickgelu_'): default_cfgs[n] for n in _quick_gelu_cfgs
})
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> VisionTransformer:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


# @register_model
# def refine_cfl(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Small (ViT-S/16)
#     """
#     model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
#     model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model

# @register_model
# def raw_vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model
