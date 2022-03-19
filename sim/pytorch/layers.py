#! -*- coding: utf-8 -*-
""" Pytorch Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
from sim.pytorch.common import get_activation
from sim.pytorch.common import scaled_dot_product_attention
from sim.pytorch.common import truncated_normal_
from typing import Any


class PositionEmbedding(nn.Module):
    """定义可训练的位置Embedding
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 merge_mode: str = "add",
                 hierarchical: Any = None,
                 custom_position_ids: bool = False,
                 initializer: Any = truncated_normal_(),
                 device: Any = None,
                 dtype: Any = None):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param merge_mode: 输入和position合并的方式
        :param hierarchical: 是否层次分解位置编码
        :param custom_position_ids: 是否传入自定义位置编码id
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.custom_position_ids = custom_position_ids
        self.initializer = initializer
        self.weight = nn.Parameter(torch.empty((self.input_dim, self.output_dim), **factory_kwargs))
        self.initializer(self.weight)

    def forward(self, inputs):
        """如果传入自定义position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            position_ids = position_ids.int()
        else:
            batch_size, seq_len = inputs.size()[0], inputs.size()[1]
            position_ids = torch.arange(start=0, end=seq_len, step=1).unsqueeze(0)

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.weight - alpha * self.weight[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = embeddings.index_select(dim=0, index=position_ids // self.input_dim)
            embeddings_y = embeddings.index_select(dim=0, index=position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = self.weight.index_select(dim=0, index=position_ids)
            else:
                embeddings = self.weight[None, :seq_len]

        if self.merge_mode == "add":
            return inputs + embeddings
        elif self.merge_mode == "mul":
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == "zero":
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = embeddings.repeat(batch_size, 1, 1)
            return torch.cat(tensors=(inputs, embeddings), dim=-1)


class RelativePositionEmbedding(nn.Module):
    """定义相对位置编码：https://arxiv.org/abs/1803.02155
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 initializer: Any = truncated_normal_(),
                 requires_grad: bool = False,
                 device: Any = None,
                 dtype: Any = None):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param initializer: 初始化器
        :param requires_grad: 是否可训练
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RelativePositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer

        self.weight = nn.Parameter(data=torch.empty((self.input_dim, self.output_dim),
                                                    requires_grad=requires_grad, **factory_kwargs))
        self.initializer(self.weight)

    def forward(self, query, value):
        # 计算位置差
        query_idx = torch.arange(0, query.shape[1])[:, None]
        value_idx = torch.arange(0, value.shape[1])[None, :]
        pos_ids = value_idx - query_idx

        max_position = (self.input_dim - 1) // 2
        pos_ids = torch.clamp(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return self.weight.index_select(dim=0, index=pos_ids)


class BertSelfAttention(nn.Module):
    """定义Self-Attention
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 batch_size: int,
                 attention_dropout: float,
                 use_bias: bool = True,
                 key_size: int = None,
                 hidden_size: int = None,
                 initializer: Any = nn.init.xavier_normal_,
                 pos_type: str = None):
        """
        :param num_heads: 注意力头数
        :param head_size: Attention中V的head_size
        :param batch_size: batch size
        :param attention_dropout: Attention矩阵的Dropout比例
        :param use_bias: 是否加上偏差项
        :param key_size: Attention中Q,K的head_size
        :param hidden_size: 编码维度
        :param initializer: 初始化器
        :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
        """
        super(BertSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.key_size = key_size if key_size is not None else head_size
        self.hidden_size = hidden_size if hidden_size is not None else num_heads * head_size
        self.initializer = initializer
        self.pos_type = pos_type

        self.query_dense = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.key_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.query_dense.weight)
        self.key_dense = nn.Linear(in_features=self.hidden_size,
                                   out_features=self.key_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.key_dense.weight)
        self.value_dense = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.head_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.value_dense.weight)
        self.output_dense = nn.Linear(in_features=self.head_size * self.num_heads,
                                      out_features=self.hidden_size, bias=self.use_bias)
        self.initializer(self.output_dense.weight)

    def transpose_for_scores(self, input_tensor: Any, head_size: int):
        """分拆最后一个维度到 (num_heads, depth)
        :param input_tensor: 输入
        :param head_size: 每个注意力头维数
        """
        input_tensor = torch.reshape(input=input_tensor, shape=(self.batch_size, -1, self.num_heads, head_size))
        return input_tensor.permute(0, 2, 1, 3)

    def forward(self, inputs):
        pos_ids = None
        if self.pos_type == "typical_relation":
            query, key, value, pos_ids, mask = inputs
        else:
            query, key, value, mask = inputs

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.transpose_for_scores(input_tensor=query, head_size=self.key_size)
        key = self.transpose_for_scores(input_tensor=key, head_size=self.key_size)
        value = self.transpose_for_scores(input_tensor=value, head_size=self.head_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            attention_head_size=self.head_size,
            dropout=self.attention_dropout,
            mask=mask,
            pos_type=self.pos_type,
            pos_ids=pos_ids
        )

        attn_outputs = self.output_dense(scaled_attention)

        return attn_outputs, attention_weights


class Embedding(nn.Embedding):
    """扩展Embedding层
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None,
                 max_norm: float = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: torch.Tensor = None,
                 device=None, dtype=None) -> None:
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                                        norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)

    def forward(self, inputs: torch.Tensor, mode: str = "embedding") -> torch.Tensor:
        """新增mode参数，可以为embedding或dense。如果为embedding，
           则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == "embedding":
            return F.embedding(inputs, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(input=inputs, other=self.weight.permute(1, 0))


class BiasAdd(nn.Module):
    """偏置项
    """

    def __init__(self, shape: tuple, device: Any = None, dtype: Any = None):
        """
        :param shape:
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(BiasAdd, self).__init__()
        self.weight = nn.Parameter(torch.empty(shape, **factory_kwargs))
        nn.init.zeros_(self.weight)

    def forward(self, inputs):
        return inputs + self.weight


class FeedForward(nn.Module):
    """FeedForward层
    """

    def __init__(self,
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 activation: Any = "gelu",
                 use_bias: bool = True,
                 initializer: Any = truncated_normal_()):
        """
        https://arxiv.org/abs/2002.05202
        :param in_features: 输入维度
        :param mid_features: 中间层维度
        :param out_features: 输出维度
        :param use_bias: 是否使用偏差项
        :param activation: 激活函数，如果传入的是list，则将使用门控线性单元
        :param initializer: 初始化器
        """
        super(FeedForward, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.activation = [activation] if not isinstance(activation, list) else activation
        self.use_bias = use_bias
        self.initializer = initializer

        self.input_dense = nn.Linear(in_features=self.in_features, out_features=self.mid_features, bias=self.use_bias)
        self.initializer(self.input_dense.weight)

        for index in range(1, len(self.activation)):
            setattr(self, f"inner_dense_{index}", nn.Linear(
                in_features=self.in_features, out_features=self.mid_features, bias=self.use_bias
            ))
            self.initializer(getattr(self, f"inner_dense_{index}").weight)

        self.output_dense = nn.Linear(in_features=self.mid_features, out_features=self.out_features, bias=self.use_bias)
        self.initializer(self.output_dense.weight)

    def forward(self, inputs):
        outputs = self.input_dense(inputs)
        outputs = get_activation(self.activation[0])(outputs)

        for index in range(1, len(self.activation)):
            inner_outputs = getattr(self, f"inner_dense_{index}")(inputs)
            inner_outputs = get_activation(self.activation[index])(inner_outputs)
            outputs = outputs * inner_outputs

        outputs = self.output_dense(outputs)

        return outputs


class BertOutput(nn.Module):
    """Bert 规范化输出
    """

    def __init__(self,
                 with_pool: Any = True,
                 with_nsp: Any = False,
                 with_mlm: Any = False,
                 initializer: Any = truncated_normal_(),
                 hidden_size: int = None,
                 embedding_size: int = None,
                 hidden_act: str = None,
                 layer_norm_eps: float = None,
                 mlm_decoder: Any = None,
                 mlm_decoder_arg: dict = None,
                 vocab_size: int = None):
        """
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        :param initializer: 初始化器
        :param hidden_size: 隐藏层大小
        :param embedding_size: 词嵌入大小
        :param hidden_act: encoder和pool中的非线性激活函数
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param mlm_decoder: 用于给mlm做vocab分类的层，可训练，相当于无bias的dense
        :param
        :param vocab_size: mlm_decoder必传
        """
        super(BertOutput, self).__init__()
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = initializer
        self.hidden_size = hidden_size

        if self.with_pool:
            self.pool_activation = {"act": "tanh", "arg": {}} if with_pool is True else with_pool
            self.pooler_dense = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.initializer(self.pooler_dense.weight)

            if self.with_nsp:
                self.nsp_prob = nn.Linear(in_features=self.hidden_size, out_features=2)
                self.initializer(self.nsp_prob.weight)

        if self.with_mlm:
            self.mlm_activation = {"act": "softmax", "arg": {"dim": -1}} if with_mlm is True else with_mlm
            self.embedding_size = embedding_size
            self.hidden_act = hidden_act
            self.layer_norm_eps = layer_norm_eps
            self.mlm_decoder = mlm_decoder
            self.mlm_decoder_arg = {} if mlm_decoder_arg is None else mlm_decoder_arg

            self.mlm_dense = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)
            self.initializer(self.mlm_dense.weight)
            self.mlm_norm = nn.LayerNorm(normalized_shape=self.embedding_size, eps=self.layer_norm_eps)
            self.mlm_bias = BiasAdd(shape=(vocab_size,))

    def forward(self, inputs):
        outputs = []
        if self.with_pool:
            sub_outputs = inputs[:, 0]
            sub_outputs = self.pooler_dense(sub_outputs)
            sub_outputs = get_activation(self.pool_activation["act"])(sub_outputs, **self.pool_activation["arg"])

            if self.with_nsp:
                sub_outputs = self.nsp_prob(sub_outputs)
                sub_outputs = get_activation("softmax")(sub_outputs, dim=-1)
            outputs.append(sub_outputs)

        if self.with_mlm:
            sub_outputs = self.mlm_dense(inputs)
            sub_outputs = get_activation(self.hidden_act)(sub_outputs)
            sub_outputs = self.mlm_norm(sub_outputs)(sub_outputs)
            sub_outputs = self.mlm_decoder(sub_outputs, **self.mlm_decoder_arg)
            sub_outputs = self.mlm_bias(sub_outputs)
            sub_outputs = get_activation(self.mlm_activation["act"])(sub_outputs, **self.mlm_activation["arg"])
            outputs.append(sub_outputs)

        if not outputs:
            return inputs
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__()
        self.p = p
        self.noise_shape = None

    def forward(self, inputs, noise_shape=None):
        """ noise_shape，tuple，应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """

        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.p == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.p == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.p).div_(1 - self.p)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)
