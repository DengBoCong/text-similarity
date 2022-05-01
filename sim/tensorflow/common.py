#! -*- coding: utf-8 -*-
""" Tensorflow Common Tools
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras as keras
from distutils.util import strtobool
from sim.tools.tools import orthogonally_resize
from tensorflow.python.eager import tape
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from typing import Any
from typing import NoReturn

# 是否启动重计算
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def set_seed(manual_seed: int):
    """固定随机种子
    :param manual_seed: 随机种子
    """
    random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    np.random.seed(manual_seed)
    tf.random.set_seed(manual_seed)


def load_embeddings(embeddings: np.ndarray, keep_tokens: list = None, compound_tokens: list = None) -> Any:
    """给加载与训练权重时，token不一致进行额外处理用
    :param embeddings: 原全量embedding
    :param keep_tokens: 要保留的词ID列表
    :param compound_tokens: 扩展Embedding
    """
    if keep_tokens is not None:
        embeddings = tf.gather(params=embeddings, indices=keep_tokens)

    if compound_tokens is not None:
        ext_embeddings = []
        for item in compound_tokens:
            if isinstance(item, list):
                item = (item, [1] * len(item))

            ext_embeddings.append(np.average(tf.gather(embeddings, item[0]), 0, item[1]))

        embeddings = np.concatenate(arrays=[embeddings, ext_embeddings], axis=0)

    return embeddings


def load_checkpoint(checkpoint_dir: str,
                    execute_type: str,
                    checkpoint_save_size: int,
                    model: tf.keras.Model = None,
                    encoder: tf.keras.Model = None,
                    decoder: tf.keras.Model = None) -> tf.train.CheckpointManager:
    """加载检查点，同时支持Encoder-Decoder结构加载，两种类型的模型二者只能传其一
    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    :param model: 传入的模型
    :param encoder: 传入的Encoder模型
    :param decoder: 传入的Decoder模型
    """
    if model is not None:
        checkpoint = tf.train.Checkpoint(model=model)
    elif encoder is not None and decoder is not None:
        checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    else:
        raise ValueError("Create checkpoint error")

    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir,
                                                    max_to_keep=checkpoint_save_size)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    else:
        if execute_type != "train" and execute_type != "pre_treat":
            raise FileNotFoundError("Not found checkpoint file")

    return checkpoint_manager


def load_bert_weights_from_checkpoint(checkpoint: Any, model: keras.Model, mapping: dict) -> NoReturn:
    """ 根据mapping从checkpoint加载bert权重
    :param checkpoint: 检查点，str/dict
    :param model: 模型
    :param mapping: 权重映射表
    """
    weight_value_pairs, weights, values = [], [], []
    for trainable_weight in model.trainable_weights:
        try:
            weight_name = trainable_weight.name.split(":")[0]
            if weight_name not in mapping:
                continue

            if isinstance(checkpoint, dict):
                variable = checkpoint[mapping[weight_name]]
            else:
                variable = tf.train.load_variable(checkpoint, mapping[weight_name])

            if mapping[weight_name] in [
                "bert/embeddings/word_embeddings",
                "cls/predictions/output_bias"
            ]:
                values.append(load_embeddings(variable))
            elif mapping[weight_name] == "cls/seq_relationship/output_weights":
                values.append(tf.transpose(a=variable, perm=[1, 0]))
            else:
                values.append(variable)
            weights.append(trainable_weight)
        except Exception as e:
            print(f"{str(e)}, but ignored")

    for weight, value in zip(weights, values):
        if value is not None:
            weight_shape, value_shape = weight.shape, value.shape
            if weight_shape != value_shape:
                value = orthogonally_resize(value, weight_shape)

            weight_value_pairs.append((weight, value))

    keras.backend.batch_set_value(weight_value_pairs)


def load_weights_from_checkpoint(checkpoint: Any, model: keras.Model, mapping: dict = None) -> NoReturn:
    """通用模型权重加载
    :param checkpoint: 检查点，str/dict
    :param model: 模型
    :param mapping: 权重映射表，为空的话就按照模型本身结构名加载权重
    """
    weight_value_pairs, weights, values = [], [], []
    for trainable_weight in model.trainable_weights:
        try:
            weight_name = trainable_weight.name.split(":")[0]

            if mapping and weight_name not in mapping:
                print(f"`{weight_name}` not in weights mapping, and ignore")
                continue

            if isinstance(checkpoint, dict):
                variable = checkpoint[mapping[weight_name] if mapping else weight_name]
            else:
                variable = tf.train.load_variable(checkpoint, mapping[weight_name] if mapping else weight_name)

            values.append(variable)
            weights.append(trainable_weight)
        except Exception as e:
            print(f"{str(e)}, but ignored")

    for weight, value in zip(weights, values):
        if value is not None:
            weight_shape, value_shape = weight.shape, value.shape
            if weight_shape != value_shape:
                raise ValueError(f"shape {weight_shape} and shape {value_shape} are incompatible")

            weight_value_pairs.append((weight, value))

    keras.backend.batch_set_value(weight_value_pairs)


# 定义相关的损失函数
def contrastive_loss(ew: Any, label: Any, m: float):
    """
    :param ew: Embedding向量之间的度量
    :param label: 样本句子的标签
    :param m: 负样本控制阈值
    :return:
    """
    l_1 = 0.25 * (1.0 - ew) * (1.0 - ew)
    l_0 = tf.where(condition=ew < m * tf.ones_like(input=ew), x=tf.fill(dims=ew.shape, value=0), y=ew) * tf.where(
        condition=ew < m * tf.ones_like(input=ew), x=tf.fill(dims=ew.shape, value=0), y=ew)

    loss = label * 1.0 * l_1 + (1 - label) * 1.0 * l_0

    return loss.sum()


def scaled_dot_product_attention(query: tf.Tensor,
                                 key: tf.Tensor,
                                 value: tf.Tensor,
                                 batch_size: int,
                                 num_heads: int,
                                 attention_head_size: int,
                                 dropout: float,
                                 mask: Any = None,
                                 pos_type: str = None,
                                 pos_ids: Any = None) -> tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param batch_size: batch size
    :param num_heads: head num
    :param attention_head_size: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
    :param pos_ids: 位置编码
    """
    attention_scores = tf.matmul(a=query, b=key, transpose_b=True)
    # 处理位置编码
    if pos_type == "typical_relation":
        attention_scores = attention_scores + tf.einsum("bhjd,kjd->bhjk", query, pos_ids)
    attention_scores = attention_scores / tf.math.sqrt(x=tf.cast(x=attention_head_size, dtype="float32"))

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
    attention_weights = keras.layers.Dropout(rate=dropout)(attention_weights)

    context_layer = tf.matmul(a=attention_weights, b=value)
    if pos_type == "typical_relation":
        context_layer = context_layer + tf.einsum("bhjk,jkd->bhjd", attention_weights, pos_ids)
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, attention_head_size * num_heads))

    return context_layer, attention_weights


def dot_product_attention(query: tf.Tensor,
                          key: tf.Tensor,
                          value: tf.Tensor,
                          depth: int,
                          dropout: float,
                          mask: Any = None) -> tuple:
    """通用点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param depth: 特征维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    """
    attention_scores = tf.matmul(a=query, b=key, transpose_b=True)
    attention_scores = attention_scores / tf.math.sqrt(x=tf.cast(x=depth, dtype="float32"))

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
    attention_weights = keras.layers.Dropout(rate=dropout)(attention_weights)
    context_layer = tf.matmul(a=attention_weights, b=value)

    return context_layer, attention_weights


# 参考自 https://github.com/bojone/keras_recompute
def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            with tape.stop_recording():
                outputs = kernel_call()
                outputs = tf.identity(outputs)

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(outputs, watches, output_gradients=[doutputs])
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        outputs, grad_fn = call_and_grad(*flat_inputs)
        flat_outputs = nest.flatten(outputs)

        def actual_grad_fn(*doutputs):
            grads = grad_fn(*doutputs, variables=self.trainable_weights)
            return grads[0] + grads[1]

        watches = flat_inputs + self.trainable_weights
        watches = [tf.convert_to_tensor(x) for x in watches]
        tape.record_operation(
            call.__name__, flat_outputs, watches, actual_grad_fn
        )
        return outputs

    return inner


def get_angles(pos: Any, i: Any, depth: Any):
    """pos/10000^(2i/d_model)
    :param pos: 字符总的数量按顺序递增
    :param i: 词嵌入大小按顺序递增
    :param depth: 位置嵌入大小
    :return: shape=(pos.shape[0], d_model)
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))
    return pos * angle_rates


class Sinusoidal(keras.initializers.Initializer):
    """Sin-Cos位置向量初始化器
    来自：https://arxiv.org/abs/1706.03762
    """

    def __call__(self, shape: Any, dtype: Any = None, *args, **kwargs):
        """Sin-Cos形式的位置向量
        """
        position, depth = shape
        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(depth)[np.newaxis, :], depth)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return angle_rads


def sparse_dropout(x: Any, rate: float, noise_shape: Any) -> Any:
    """Sparse Dropout
    :param x: 输入
    :param rate: 采样率
    :param noise_shape: 引入noise shape
    """
    random_tensor = tf.random.uniform(shape=noise_shape) + rate
    dropout_mask = tf.cast(x=tf.floor(x=random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(sp_input=x, to_retain=dropout_mask)
    return pre_out * (1. / rate)
