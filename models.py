import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
import nn
import math


def model_arg_scope(**kwargs):
    """Create new counter and apply arg scope to all arg scoped nn
    operations."""
    counters = {}
    return arg_scope(
            [nn.conv2d, nn.deconv2d, nn.residual_block, nn.dense, nn.activate],
            counters = counters, **kwargs)


def dec_up(
        c, init = False, dropout_p = 0.5,
        n_scales = 1, n_residual_blocks = 2, activation = "elu", n_filters = 64, max_filters = 256):
    with model_arg_scope(
            init = init, dropout_p = dropout_p, activation = activation):
        # outputs
        hs = []
        # prepare input
        h = nn.nin(c, n_filters)
        for l in range(n_scales):
            # level module
            for i in range(n_residual_blocks):
                h = nn.residual_block(h)
                hs.append(h)
            # prepare input to next level
            if l + 1 < n_scales:
                n_filters = min(2*n_filters, max_filters)
                h = nn.downsample(h, n_filters)
        return hs


def dec_down(
        gs, init = False, dropout_p = 0.5,
        n_scales = 1, n_residual_blocks = 2, activation = "elu"):
    gs = list(gs)
    with model_arg_scope(
            init = init, dropout_p = dropout_p, activation = activation):
        # outputs
        hs = []
        # prepare input
        n_filters = gs[-1].shape.as_list()[-1]
        h = nn.nin(gs[-1], n_filters)
        for l in range(n_scales):
            # level module
            for i in range(n_residual_blocks):
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            # prepare input to next level
            if l + 1 < n_scales:
                n_filters = gs[-1].shape.as_list()[-1]
                h = nn.upsample(h, n_filters)

        return hs


def dec_params(
        x, h, init = False, **kwargs):
    with model_arg_scope(init = init):
        num_filters = x.shape.as_list()[-1]
        return nn.conv2d(h, num_filters)


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.make_template(name, run, unique_name_ = name)
