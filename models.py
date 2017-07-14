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


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.make_template(name, run, unique_name_ = name)


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
        gs, zs_posterior, training, init = False, dropout_p = 0.5,
        n_scales = 1, n_residual_blocks = 2, activation = "elu"):
    assert n_residual_blocks % 2 == 0
    gs = list(gs)
    zs_posterior = list(zs_posterior)
    with model_arg_scope(
            init = init, dropout_p = dropout_p, activation = activation):
        # outputs
        hs = [] # hidden units
        ps = [] # priors
        zs = [] # prior samples
        # prepare input
        n_filters = gs[-1].shape.as_list()[-1]
        h = nn.nin(gs[-1], n_filters)
        for l in range(n_scales):
            # level module
            ## hidden units
            for i in range(n_residual_blocks // 2):
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            ## prior
            p = latent_parameters(h)
            ps.append(p)
            ## prior sample
            z_prior = latent_sample(p)
            zs.append(z_prior)
            ## feedback sampled from
            if training:
                ## posterior
                z = zs_posterior.pop(0)
            else:
                ## prior
                z = z_prior
            for i in range(n_residual_blocks // 2):
                n_h_channels = h.shape.as_list()[-1]
                h = tf.concat([h, z], axis = -1)
                h = nn.nin(h, n_h_channels)
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            # prepare input to next level
            if l + 1 < n_scales:
                n_filters = gs[-1].shape.as_list()[-1]
                h = nn.upsample(h, n_filters)

        assert not gs
        if training:
            assert not zs_posterior

        return hs, ps, zs


def enc_up(
        x, c, init = False, dropout_p = 0.5,
        n_scales = 1, n_residual_blocks = 2, activation = "elu", n_filters = 64, max_filters = 256):
    with model_arg_scope(
            init = init, dropout_p = dropout_p, activation = activation):
        # outputs
        hs = []
        # prepare input
        xc = tf.concat([x,c], axis = -1)
        h = nn.nin(xc, n_filters)
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


def enc_down(
        gs, init = False, dropout_p = 0.5,
        n_scales = 1, n_residual_blocks = 2, activation = "elu"):
    assert n_residual_blocks % 2 == 0
    gs = list(gs)
    with model_arg_scope(
            init = init, dropout_p = dropout_p, activation = activation):
        # outputs
        hs = [] # hidden units
        qs = [] # posteriors
        zs = [] # samples from posterior
        # prepare input
        n_filters = gs[-1].shape.as_list()[-1]
        h = nn.nin(gs[-1], n_filters)
        for l in range(n_scales):
            # level module
            ## hidden units
            for i in range(n_residual_blocks // 2):
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            ## posterior parameters
            q = latent_parameters(h)
            qs.append(q)
            ## posterior sample
            z = latent_sample(q)
            zs.append(z)
            ## sample feedback
            for i in range(n_residual_blocks // 2):
                gz = tf.concat([gs.pop(), z], axis = -1)
                h = nn.residual_block(h, gz)
                hs.append(h)
            # prepare input to next level
            if l + 1 < n_scales:
                n_filters = gs[-1].shape.as_list()[-1]
                h = nn.upsample(h, n_filters)

        assert not gs

        return hs, qs, zs


# Distributions


def dec_parameters(
        h, init = False, **kwargs):
    with model_arg_scope(init = init):
        num_filters = 3
        return nn.conv2d(h, num_filters)


def dec_sample(p):
    return p


def latent_parameters(
        h, init = False, **kwargs):
    num_filters = h.shape.as_list()[-1]
    return nn.conv2d(h, 2*num_filters)


def logvarvar(u):
    cutoff = tf.to_float(-5)
    logvar = tf.maximum(cutoff, u)
    var = tf.exp(logvar)
    return logvar, var


def latent_sample(p):
    mean, u = tf.split(p, 2, axis = 3)
    logvar, var = logvarvar(u)
    stddev = tf.sqrt(var)
    eps = tf.random_normal(mean.shape, mean = 0.0, stddev = 1.0)
    return mean + stddev * eps


def latent_kl(q, p):
    mean1, u1 = tf.split(q, 2, axis = 3)
    logvar1, var1 = logvarvar(u1)
    mean2, u2 = tf.split(p, 2, axis = 3)
    logvar2, var2 = logvarvar(u2)

    kl = 0.5*(var1/var2 - 1.0 + tf.square(mean2 - mean1) / var2 + logvar2 - logvar1)
    kl = tf.reduce_sum(kl, axis = [1,2,3])
    kl = tf.reduce_mean(kl)
    return kl
