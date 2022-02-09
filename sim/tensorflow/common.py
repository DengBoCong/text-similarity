#! -*- coding: utf-8 -*-
""" Tensorflow Common Tools
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.keras as keras
from distutils.util import strtobool
from tensorflow.python.eager import tape
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from typing import Any

# 是否启动重计算
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


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


def load_bert_weights_from_checkpoint(checkpoint_path: str, model: keras.Model, mapping: dict):
    """ 根据mapping从checkpoint加载bert权重
    :param checkpoint_path: 检查点路径，不同于load_checkpoint中的dir
    :param model: 模型
    :param mapping: 权重映射表
    """
    mapping = {k: v for k, v in mapping.items() if k in layers_name}

    weight_value_pairs = []
    # for layer, variables in mapping.items():
    #     layer =


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
                                 hidden_size: int,
                                 attention_head_size: int,
                                 dropout: float,
                                 mask: Any = None) -> tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param batch_size: batch size
    :param hidden_size: hidden size
    :param attention_head_size: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    """
    attention_scores = tf.matmul(a=query, b=key, transpose_b=True)
    attention_scores = attention_scores / tf.math.sqrt(x=tf.cast(x=attention_head_size, dtype="float32"))

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
    attention_weights = keras.layers.Dropout(rate=dropout)(attention_weights)

    context_layer = tf.matmul(a=attention_weights, b=value)
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, hidden_size))

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
