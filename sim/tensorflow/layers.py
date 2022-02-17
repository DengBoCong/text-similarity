#! -*- coding: utf-8 -*-
""" Tensorflow Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from sim.tensorflow.common import recompute_grad
from sim.tensorflow.common import scaled_dot_product_attention
from typing import Any


class PositionEmbedding(keras.layers.Layer):
    """定义可训练的位置Embedding
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 merge_mode: str = "add",
                 hierarchical: Any = None,
                 custom_position_ids: bool = False,
                 embeddings_initializer: Any = "zeros",
                 **kwargs):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param merge_mode: 输入和position合并的方式
        :param hierarchical: 是否层次分解位置编码
        :param custom_position_ids: 是否传入自定义位置编码id
        :param embeddings_initializer: 初始化器
        """
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.custom_position_ids = custom_position_ids
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs, *args, **kwargs):
        """如果传入自定义position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if "int" not in position_ids.dtype.name:
                position_ids = tf.cast(x=position_ids, dtype=tf.int32)
        else:
            batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
            position_ids = tf.expand_dims(input=tf.range(start=0, limit=seq_len, delta=1), axis=0)

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = tf.gather(params=embeddings, indices=position_ids // self.input_dim)
            embeddings_y = tf.gather(params=embeddings, indices=position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = tf.gather(params=self.embeddings, indices=position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == "add":
            return inputs + embeddings
        elif self.merge_mode == "mul":
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == "zero":
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = tf.tile(input=embeddings, multiples=[batch_size, 1, 1])
            return tf.concat(values=[inputs, embeddings], axis=-1)

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "merge_model": self.merge_mode,
            "hierarchical": self.hierarchical,
            "embeddings_initializer": keras.initializers.serialize(initializer=self.embeddings_initializer),
            "custom_position_ids": self.custom_position_ids
        }
        base_config = super(PositionEmbedding, self).get_config()
        base_config.update(config)
        return base_config


class RelativePositionEmbedding(keras.layers.Layer):
    """定义相对位置编码：https://arxiv.org/abs/1803.02155
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 embeddings_initializer: Any = "zeros",
                 **kwargs):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param embeddings_initializer: 初始化器
        """
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.supports_masking = True

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs, *args, **kwargs):
        query, value = inputs
        # 计算位置差
        query_idx = keras.backend.arange(0, query.shape[1], dtype="int32")
        query_idx = tf.expand_dims(query_idx, axis=1)
        value_idx = keras.backend.arange(0, value.shape[1], dtype="int32")
        value_idx = tf.expand_dims(value_idx, axis=0)
        pos_ids = value_idx - query_idx

        max_position = (self.input_dim - 1) // 2
        pos_ids = tf.clip_by_value(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return tf.gather(params=self.embeddings, indices=pos_ids)

    def compute_output_shape(self, input_shape):
        return None, None, self.output_dim

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": keras.initializers.serialize(initializer=self.embeddings_initializer)
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        config.update(base_config)
        return config


class Embedding(keras.layers.Embedding):
    """扩展Embedding层
    """

    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if keras.backend.ndim(inputs) == 2:
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = keras.backend.ones_like(mask[:, :1], dtype="bool")
                mask2 = mask[:, 1:]
                return keras.backend.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode: str = "embedding"):
        """新增mode参数，可以为embedding或dense。如果为embedding，
           则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == "embedding":
            return super(Embedding, self).call(inputs)
        else:
            return tf.linalg.matmul(a=inputs, b=self.embeddings, transpose_b=True)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (keras.backend.int_shape(self.embeddings)[0],)


class BiasAdd(keras.layers.Layer):
    """偏置项
    """

    def __init__(self, **kwargs):
        super(BiasAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        self.bias = self.add_weight(name="bias", shape=(input_shape[-1],), initializer="zeros")

    def call(self, inputs, *args, **kwargs):
        return keras.backend.bias_add(inputs, self.bias)


class FeedForward(keras.layers.Layer):
    """FeedForward层
    """

    def __init__(self,
                 units: int,
                 activation: Any = "gelu",
                 use_bias: bool = True,
                 kernel_initializer: Any = "glorot_uniform",
                 **kwargs):
        """
        https://arxiv.org/abs/2002.05202
        :param units: 输出维度
        :param use_bias: 是否使用偏差项
        :param activation: 激活函数，如果传入的是list，则将使用门控线性单元
        :param kernel_initializer: 初始化器
        :param name: 模型名
        """
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = [activation] if not isinstance(activation, list) else activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        for index in range(len(self.activation)):
            setattr(self, f"inner_dense_{index}", keras.layers.Dense(
                units=self.units,
                activation=self.activation[index],
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                name="input" if index == 0 else f"inner-dense-{index}"
            ))

        self.output_dense = keras.layers.Dense(
            units=input_shape[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="output"
        )

    @recompute_grad
    def call(self, inputs, *args, **kwargs):
        outputs = self.inner_dense_0(inputs)
        for index in range(1, len(self.activation)):
            outputs = outputs * getattr(self, f"inner_dense_{index}")(inputs)

        outputs = self.output_dense(outputs)

        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [keras.activations.serialize(act) for act in self.activation],
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        base_config.update(config)
        return base_config


class BertSelfAttention(keras.layers.Layer):
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
                 initializer: Any = "glorot_uniform",
                 pos_type: str = None,
                 **kwargs):
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
        super(BertSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.key_size = key_size if key_size is not None else head_size
        self.hidden_size = hidden_size if hidden_size is not None else num_heads * head_size
        self.initializer = initializer
        self.pos_type = pos_type

    def build(self, input_shape):
        super(BertSelfAttention, self).build(input_shape)
        self.query_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="query")
        self.key_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                            kernel_initializer=self.initializer, name="key")
        self.value_dense = keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="value")
        self.output_dense = keras.layers.Dense(units=self.hidden_size, use_bias=self.use_bias,
                                               kernel_initializer=self.initializer, name="output")

    def transpose_for_scores(self, input_tensor: tf.Tensor, head_size: int):
        """分拆最后一个维度到 (num_heads, depth)
        :param input_tensor: 输入
        :param head_size: 每个注意力头维数
        """
        input_tensor = tf.reshape(tensor=input_tensor, shape=(self.batch_size, -1, self.num_heads, head_size))
        return tf.transpose(input_tensor, perm=[0, 2, 1, 3])

    @recompute_grad
    def call(self, inputs, *args, **kwargs):
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

    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "attention_dropout": self.attention_dropout,
            "use_bias": self.use_bias,
            "key_size": self.key_size,
            "hidden_size": self.hidden_size,
            "initializer": keras.initializers.serialize(initializer=self.initializer),
        }
        base_config = super(BertSelfAttention, self).get_config()
        base_config.update(config)
        return base_config


class BertOutput(keras.layers.Layer):
    """Bert 规范化输出
    """

    def __init__(self,
                 with_pool: Any = True,
                 with_nsp: Any = False,
                 with_mlm: Any = False,
                 initializer: Any = None,
                 hidden_size: int = None,
                 embedding_size: int = None,
                 hidden_act: str = None,
                 layer_norm_eps: float = None,
                 vocab_dense_layer: Any = None,
                 **kwargs):
        """
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, vocab_dense_layer
        :param initializer: 初始化器
        :param hidden_size: 隐藏层大小
        :param embedding_size: 词嵌入大小
        :param hidden_act: encoder和pool中的非线性激活函数
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param vocab_dense_layer: 用于给mlm做vocab分类的层，可训练，相当于无bias的dense
        """
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = keras.initializers.TruncatedNormal(stddev=0.02) if initializer is None else initializer

        if self.with_pool:
            self.pool_activation = 'tanh' if with_pool is True else with_pool
            self.hidden_size = hidden_size

        if self.with_mlm:
            self.mlm_activation = 'softmax' if with_mlm is True else with_mlm
            self.embedding_size = embedding_size
            self.hidden_act = hidden_act
            self.layer_norm_eps = layer_norm_eps
            self.vocab_dense_layer = vocab_dense_layer

        super(BertOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BertOutput, self).build(input_shape)
        if self.with_pool:
            self.pooler = keras.layers.Lambda(lambda x: x[:, 0], name="pooler")
            self.pooler_dense = keras.layers.Dense(units=self.hidden_size, activation=self.pool_activation,
                                                   kernel_initializer=self.initializer,
                                                   name="pooler-dense")
            if self.with_nsp:
                self.nsp_prob = keras.layers.Dense(units=2, activation="softmax", kernel_initializer=self.initializer,
                                                   name="nsp-prob")

        if self.with_mlm:
            self.mlm_dense = keras.layers.Dense(units=self.embedding_size, activation=self.hidden_act,
                                                kernel_initializer=self.initializer, name="mlm-dense")
            self.mlm_norm = keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name="mlm-norm")
            self.mlm_bias = BiasAdd(name="mlm-bias")
            self.mlm_act = keras.layers.Activation(activation=self.mlm_activation, name="mlm-activation")

    def call(self, inputs, *args, **kwargs):
        outputs = []
        if self.with_pool:
            sub_outputs = self.pooler(inputs)
            sub_outputs = self.pooler_dense(sub_outputs)

            if self.with_nsp:
                sub_outputs = self.nsp_prob(sub_outputs)
            outputs.append(sub_outputs)

        if self.with_mlm:
            sub_outputs = self.mlm_dense(inputs)
            sub_outputs = self.mlm_norm(sub_outputs)
            sub_outputs = self.vocab_dense_layer(sub_outputs, mode="dense")
            sub_outputs = self.mlm_bias(sub_outputs)
            sub_outputs = self.mlm_act(sub_outputs)
            outputs.append(sub_outputs)

        if not outputs:
            return inputs
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
