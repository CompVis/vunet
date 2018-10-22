"""
modified from pixelcnn++
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def int_shape(x):
    return x.shape.as_list()


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def dense(x, num_units, init_scale=1., counters={}, init=False, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        xs = x.shape.as_list()
        V = tf.get_variable('V', [xs[1], num_units], tf.float32, tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', [num_units], dtype=tf.float32, initializer=tf.constant_initializer(1.))
        b = tf.get_variable('b', [num_units], dtype=tf.float32, initializer=tf.constant_initializer(0.))

        V_norm = tf.nn.l2_normalize(V, [0])
        x = tf.matmul(x, V_norm)
        if init:
            mean, var = tf.nn.moments(x, [0])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, num_units])*x + tf.reshape(b, [1, num_units])

        return x


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, **kwargs):
    ''' convolutional layer '''
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        xs = x.shape.as_list()
        V = tf.get_variable('V', filter_size + [xs[-1], num_filters],
                            tf.float32, tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.))
        b = tf.get_variable('b', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.))

        V_norm = tf.nn.l2_normalize(V, [0,1,2])
        x = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
        if init:
            mean, var = tf.nn.moments(x, [0,1,2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters])*x + tf.reshape(b, [1, 1, 1, num_filters])

        return x


@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, **kwargs):
    ''' transposed convolutional layer '''
    num_filters = int(num_filters)
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    strides = [1] + stride + [1]
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] -
                        1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
    with tf.variable_scope(name):
        V = tf.get_variable('V',
                filter_size + [num_filters, xs[-1]],
                tf.float32,
                tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.))
        b = tf.get_variable('b', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.))

        V_norm = tf.nn.l2_normalize(V, [0,1,3])
        x = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1] + stride + [1], pad)
        if init:
            mean, var = tf.nn.moments(x, [0,1,2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters])*x + tf.reshape(b, [1, 1, 1, num_filters])

        return x


@add_arg_scope
def activate(x, activation, **kwargs):
    if activation == None:
        return x
    elif activation == "elu":
        return tf.nn.elu(x)
    else:
        raise NotImplemented(activation)


def nin(x, num_units):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units)
    return tf.reshape(x, s[:-1] + [num_units])


def downsample(x, num_units):
    return conv2d(x, num_units, stride = [2, 2])


def upsample(x, num_units, method = "subpixel"):
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride = [2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4*num_units)
        x = tf.depth_to_space(x, 2)
        return x


@add_arg_scope
def residual_block(x, a = None, conv=conv2d, init=False, dropout_p=0.0, gated = False, **kwargs):
    """Slight variation of original."""
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if a is not None:
        a = nin(activate(a), num_filters)
        residual = tf.concat([residual, a], axis = -1)
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob = 1.0 - dropout_p)
    residual = conv(residual, num_filters)
    if gated:
        residual = activate(residual)
        residual = tf.nn.dropout(residual, keep_prob = 1.0 - dropout_p)
        residual = conv(residual, 2*num_filters)
        a, b = tf.split(residual, 2, 3)
        residual = a * tf.nn.sigmoid(b)

    return x + residual


def make_linear_var(
        step,
        start, end,
        start_value, end_value,
        clip_min = 0.0, clip_max = 1.0):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha)/(b - a) * (x - a) + alpha"""
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def split_groups(x, bs = 2):
    return tf.split(tf.space_to_depth(x, bs), bs**2, axis = 3)


def merge_groups(xs, bs = 2):
    return tf.depth_to_space(tf.concat(xs, axis = 3), bs)
