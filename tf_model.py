import numpy as np # type: ignore
import tensorflow as tf
from typing import NamedTuple, Any

from gumbel_softmax import gumbel_softmax

ModuleConfig = NamedTuple('ModuleConfig', [
        ('cell_type', Any),
        ('state_size', int),
        ('dropout_prob', float),
        ('num_layers', int)
    ])

ModelConfig = NamedTuple('ModelConfig', [
        ('processing_config', ModuleConfig),
        ('batch_size', int),
        ('num_timesteps', int),
        ('vocab_size', int),
        ('movement_step_size', float)
    ])

def get_processing_module(observations, module_scope):
    """
    Returns a multi-layer RNN that goes over the observations
    num_steps is time_horizon
    """
    with tf.variable_scope(module_scope) as scope:
        cell = config.processing.cell_type(config.processing.state_size)
        cell = tf.nn.dropout(cell, keep_prob=config.processing.dropout_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.processing.num_layers)
        init_state = cell.zero_state(config.batch_size, tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell, observations, initial_state=init_state)

    return output

